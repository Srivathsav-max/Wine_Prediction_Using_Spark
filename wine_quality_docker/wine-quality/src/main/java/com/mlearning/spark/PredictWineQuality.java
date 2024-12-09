package com.mlearning.spark;

import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * This class handles loading a trained Logistic Regression model from S3,
 * performing predictions on a test dataset, and evaluating the model's performance.
 */
public class PredictWineQuality {

    public static final Logger logger = LogManager.getLogger(PredictWineQuality.class);

    // Define S3 paths (replace with your actual bucket and paths)
    private static final String TESTING_DATASET = "s3a://wine-datsets/Validation_Dataset_with_Correct_Labels.csv";
    private static final String MODEL_PATH = "models/LogisticRegressionModel";
    private static final String PREDICTIONS_PATH = "data/test_predictions/";

    public static void main(String[] args) {
        // Set logging levels to reduce verbosity
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
        Logger.getLogger("com.amazonaws.auth").setLevel(Level.ERROR);

        // Initialize SparkSession with updated AWS configuration
        SparkSession spark = SparkSession.builder()
                .appName("Wine-Quality-Prediction")
                .master("local[*]")
                .config("spark.hadoop.fs.s3a.access.key", System.getenv("AWS_ACCESS_KEY_ID"))
                .config("spark.hadoop.fs.s3a.secret.key", System.getenv("AWS_SECRET_ACCESS_KEY"))
                .config("spark.hadoop.fs.s3a.endpoint", "s3." + System.getenv("AWS_REGION") + ".amazonaws.com")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                       "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .getOrCreate();

        PredictWineQuality predictor = new PredictWineQuality();
        predictor.runPrediction(spark);
    }

    /**
     * Loads the trained model, performs predictions on the test dataset, and evaluates the results.
     *
     * @param spark The SparkSession object.
     */
    public void runPrediction(SparkSession spark) {
        logger.info("Loading model from: " + MODEL_PATH);
        PipelineModel pipelineModel = PipelineModel.load(MODEL_PATH);

        logger.info("Loading test data from: " + TESTING_DATASET);
        Dataset<Row> testDf = getDataFrame(spark, true, TESTING_DATASET).cache();

        logger.info("Making predictions on test data");
        Dataset<Row> predictionDF = pipelineModel.transform(testDf).cache();

        // Save test predictions to Parquet
        logger.info("Saving test predictions to: " + PREDICTIONS_PATH);
        predictionDF.write()
            .mode("overwrite")
            .parquet(PREDICTIONS_PATH);
        logger.info("Test predictions saved successfully.");

        // Show some predictions and evaluate metrics
        predictionDF.select("features", "label", "prediction").show(5, false);
        printMetrics(predictionDF);
    }

    /**
     * Evaluates and prints various classification metrics.
     *
     * @param predictions The Dataset containing predictions and labels.
     */
    public void printMetrics(Dataset<Row> predictions) {
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");

        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy: " + accuracy);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1);

        evaluator.setMetricName("weightedPrecision");
        double precision = evaluator.evaluate(predictions);
        System.out.println("Weighted Precision: " + precision);

        evaluator.setMetricName("weightedRecall");
        double recall = evaluator.evaluate(predictions);
        System.out.println("Weighted Recall: " + recall);
    }

    /**
     * Reads a CSV file from the specified path, assembles features, and returns the prepared Dataset.
     *
     * @param spark     The SparkSession object.
     * @param transform Whether to transform the data using VectorAssembler.
     * @param path      The S3 path to the CSV file.
     * @return The prepared Dataset<Row>.
     */
    public Dataset<Row> getDataFrame(SparkSession spark, boolean transform, String path) {
        logger.info("Reading data from: " + path);
        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                //.option("sep", ";") // Uncomment if CSV uses semicolon separator
                //.option("quote", "\"") // Uncomment if needed
                .load(path);

        // Rename columns if necessary
        Dataset<Row> lblFeatureDf = df.withColumnRenamed("quality", "label")
                .select("label", "alcohol", "sulphates", "pH", "density",
                        "free sulfur dioxide", "total sulfur dioxide", "chlorides",
                        "residual sugar", "citric acid", "volatile acidity", "fixed acidity");

        lblFeatureDf = lblFeatureDf.na().drop().cache();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"alcohol", "sulphates", "pH", "density",
                        "free sulfur dioxide", "total sulfur dioxide", "chlorides",
                        "residual sugar", "citric acid", "volatile acidity", "fixed acidity"})
                .setOutputCol("features");

        if (transform) {
            lblFeatureDf = assembler.transform(lblFeatureDf).select("label", "features");
        }

        return lblFeatureDf;
    }
}
