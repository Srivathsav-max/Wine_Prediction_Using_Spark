
#!/bin/bash

# Start Spark cluster using docker-compose
docker-compose up -d

# Wait for the Spark cluster to be ready
sleep 10

# Submit the Spark job to the cluster
spark-submit \
  --master spark://localhost:7077 \
  --deploy-mode cluster \
  --driver-memory 4g \
  --executor-memory 4g \
  --executor-cores 2 \
  --num-executors 3 \
  --conf "spark.hadoop.fs.s3a.access.key=$AWS_ACCESS_KEY_ID" \
  --conf "spark.hadoop.fs.s3a.secret.key=$AWS_SECRET_ACCESS_KEY" \
  --conf "spark.hadoop.fs.s3a.endpoint=s3.$AWS_REGION.amazonaws.com" \
  --conf "spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" \
  --conf "spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider" \
  --conf "spark.hadoop.fs.s3a.path.style.access=true" \
  --conf "spark.hadoop.fs.s3a.connection.ssl.enabled=true" \
  --class com.mlearning.spark.TrainAndPersistWineQualityDataModel \
  target/wine-quality-1.0-SNAPSHOT.jar