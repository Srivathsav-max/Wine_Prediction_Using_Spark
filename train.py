import argparse
import os
import time
import json
import shutil
import boto3
import configparser
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
from botocore.exceptions import ClientError

def read_config(config_path='config.ini'):
    """Read configuration from config.ini file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return {
        'aws_access_key_id': config.get('aws', 'aws_access_key_id'),
        'aws_secret_access_key': config.get('aws', 'aws_secret_access_key'),
        'region': config.get('aws', 'region')
    }

def download_from_s3(s3_path, local_dir='data', aws_credentials=None):
    """Download file from S3 to local directory."""
    os.makedirs(local_dir, exist_ok=True)
    
    s3_path = s3_path.replace('s3://', '')
    bucket = s3_path.split('/')[0]
    key = '/'.join(s3_path.split('/')[1:])
    filename = os.path.basename(key)
    local_path = os.path.join(local_dir, filename)
    
    print(f"\nDownloading {s3_path} to {local_path}")
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_credentials['aws_access_key_id'],
            aws_secret_access_key=aws_credentials['aws_secret_access_key'],
            region_name=aws_credentials['region']
        )
        s3_client.download_file(bucket, key, local_path)
        print(f"Successfully downloaded to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading from S3: {str(e)}")
        raise

def create_spark_session(master_url=None):
    """Create a Spark session with proper configuration."""
    try:
        # Initialize builder with required configurations
        builder = SparkSession.builder \
            .appName("WineQualityTraining") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.executor.instances", "3") \
            .config("spark.default.parallelism", "12") \
            .config("spark.sql.shuffle.partitions", "12") \
            .config("spark.metrics.staticSources.enabled", "true") \
            .config("spark.metrics.conf.*.sink.servlet.class", "org.apache.spark.metrics.sink.MetricsServlet")

        # Add master URL if provided
        if master_url:
            builder = builder.master(master_url)
        else:
            builder = builder.master("local[*]")  # Use all available cores locally

        # Create and verify session
        spark = builder.getOrCreate()
        
        # Verify Spark context is running
        if not spark.sparkContext._jsc:
            raise Exception("Failed to initialize Spark context")
            
        print(f"Created Spark session with app id: {spark.sparkContext.applicationId}")
        return spark

    except Exception as e:
        print(f"Error creating Spark session: {str(e)}")
        raise

def cleanup_spark(spark):
    """Safely stop Spark session"""
    if spark and spark.sparkContext._jsc:
        try:
            spark.stop()
            print("Spark session stopped successfully")
        except Exception as e:
            print(f"Error stopping Spark: {str(e)}")

def prepare_data(spark, s3_path):
    """Load and prepare data directly from S3."""
    print(f"\nLoading data from: {s3_path}")
    start_time = time.time()
    
    try:
        # Convert s3:// to s3a:// if needed
        if s3_path.startswith('s3://'):
            s3_path = 's3a://' + s3_path[5:]
        
        # Read the data directly from S3
        data = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(s3_path)
        
        # Define feature columns (all except 'quality')
        feature_columns = [col for col in data.columns if col != 'quality']
        print(f"Using features: {', '.join(feature_columns)}")
        
        # Assemble features
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data = assembler.transform(data)
        
        # Cast label to integer
        data = data.withColumn("label", col("quality").cast(IntegerType()))
        
        print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
        print(f"Number of records: {data.count()}")
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def train_model(training_data, validation_data):
    """Train a logistic regression model."""
    print("\nInitializing model training...")
    start_time = time.time()
    
    # Initialize logistic regression with specific parameters
    lr = LogisticRegression(
        featuresCol="features", 
        labelCol="label",
        maxIter=10,
        regParam=0.3,
        elasticNetParam=0.8,
        family="multinomial"  # Explicitly set multinomial
    )
    
    # Train model
    print("Training model...")
    model = lr.fit(training_data)
    
    # Evaluate on validation data
    predictions = model.transform(validation_data)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    
    return model

def save_model_locally(model, project_dir='models'):
    """Save model in the project directory."""
    model_path = os.path.join(project_dir, 'wine_quality_model')
    
    # Ensure directory exists and is empty
    os.makedirs(project_dir, exist_ok=True)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    
    try:
        print(f"\nSaving model to: {model_path}")
        
        # Save model with all parameters
        model.write().overwrite().save(model_path)
        
        # Save additional model metadata manually
        metadata_dir = os.path.join(model_path, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata = {
            "class": "org.apache.spark.ml.classification.LogisticRegressionModel",
            "timestamp": int(time.time() * 1000),
            "sparkVersion": "3.2.0",
            "uid": model.uid,
            "paramMap": {
                "maxIter": model.getMaxIter(),
                "regParam": model.getRegParam(),
                "elasticNetParam": model.getElasticNetParam(),
                "family": "multinomial",
                "featuresCol": "features",
                "labelCol": "label",
                "predictionCol": "prediction",
                "numFeatures": model.numFeatures,
                "numClasses": model.numClasses
            }
        }
        
        # Save metadata as JSON
        metadata_file = os.path.join(metadata_dir, "part-00000")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Create _SUCCESS file in metadata directory
        success_file = os.path.join(metadata_dir, "_SUCCESS")
        open(success_file, 'a').close()
        
        # Save model parameters for reference
        params_file = os.path.join(model_path, "model_info.txt")
        with open(params_file, 'w') as f:
            f.write("Model Parameters:\n")
            f.write(f"Number of Features: {model.numFeatures}\n")
            f.write(f"Number of Classes: {model.numClasses}\n")
            f.write(f"Max Iterations: {model.getMaxIter()}\n")
            f.write(f"Regularization Parameter: {model.getRegParam()}\n")
            f.write(f"Elastic Net Parameter: {model.getElasticNetParam()}\n")
            f.write("\nTraining Information:\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print("Model and metadata saved successfully!")
        return model_path
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        raise

def verify_saved_model(model_path):
    """Verify that the model was saved correctly."""
    try:
        # Check directory structure
        required_files = [
            os.path.join(model_path, "metadata", "part-00000"),
            os.path.join(model_path, "metadata", "_SUCCESS"),
            os.path.join(model_path, "model_info.txt")
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Missing required file: {file_path}")
                return False
                
        # Verify metadata content
        metadata_file = os.path.join(model_path, "metadata", "part-00000")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        if "class" not in metadata or "paramMap" not in metadata:
            print("Invalid metadata structure")
            return False
            
        print("\nModel files verified successfully!")
        return True
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        return False
    
def upload_to_s3(local_path, s3_path, aws_credentials):
    """Upload model directory to S3."""
    s3_path = s3_path.replace('s3://', '')
    bucket = s3_path.split('/')[0]
    key_prefix = '/'.join(s3_path.split('/')[1:])
    
    print(f"\nUploading model to S3:")
    print(f"From: {local_path}")
    print(f"To: s3://{bucket}/{key_prefix}")
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_credentials['aws_access_key_id'],
            aws_secret_access_key=aws_credentials['aws_secret_access_key'],
            region_name=aws_credentials['region']
        )
        
        for root, dirs, files in os.walk(local_path):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_file_path, local_path)
                s3_key = os.path.join(key_prefix, relative_path)
                
                print(f"Uploading {relative_path}...")
                s3_client.upload_file(local_file_path, bucket, s3_key)
                
                # Verify upload
                try:
                    s3_client.head_object(Bucket=bucket, Key=s3_key)
                    print(f"Verified upload of {s3_key}")
                except ClientError:
                    raise Exception(f"Failed to verify upload of {s3_key}")
        
        print("Model uploaded to S3 successfully!")
        return True
        
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train wine quality prediction model.')
    parser.add_argument('--train_data', required=True, help='S3 path to training data')
    parser.add_argument('--validation_data', required=True, help='S3 path to validation data')
    parser.add_argument('--model_output', required=True, help='S3 path to save the trained model')
    parser.add_argument('--config', default='config.ini', help='Path to config file')
    args = parser.parse_args()
    
    print("Starting Wine Quality Prediction Model Training")
    print("=============================================")
    
    try:
        # Read AWS credentials from config
        aws_credentials = read_config(args.config)
        print("AWS credentials loaded from config.ini")
        
        # Create Spark session with AWS credentials
        spark = create_spark_session()
        
        # Configure S3A access for Spark
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        hadoop_conf.set("fs.s3a.access.key", aws_credentials['aws_access_key_id'])
        hadoop_conf.set("fs.s3a.secret.key", aws_credentials['aws_secret_access_key'])
        hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        
        # Prepare data directly from S3
        print("\nPreparing training data...")
        training_data = prepare_data(spark, args.train_data)
        
        print("\nPreparing validation data...")
        validation_data = prepare_data(spark, args.validation_data)
        
        # Train model
        model = train_model(training_data, validation_data)
        
        # Print model details before saving
        print("\nModel Details:")
        print(f"Number of Features: {model.numFeatures}")
        print(f"Number of Classes: {model.numClasses}")
        
        # Save model locally
        work_dir = os.getcwd()
        local_model_path = save_model_locally(model, os.path.join(work_dir, 'models'))
        
        # Verify saved model
        print("\nVerifying saved model...")
        if verify_saved_model(local_model_path):
            print("Model verified successfully!")
            
            # Upload model to S3
            print("\nUploading model to S3...")
            upload_to_s3(local_model_path, args.model_output, aws_credentials)
        else:
            raise Exception("Model verification failed")
        
        print("\nTraining pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        if 'spark' in locals():
            cleanup_spark(spark)

if __name__ == "__main__":
    main()