```
spark-submit   
  --class com.mlearning.spark.TrainAndPersistWineQualityDataModel
  --master spark://<master-node-ip>:7077
  --conf spark.executor.memory=3g
  --conf spark.driver.memory=3g
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem
  --conf spark.hadoop.fs.s3a.access.key=acess_key
  --conf spark.hadoop.fs.s3a.secret.key=secret_key
  --conf spark.hadoop.fs.s3a.path.style.access=true
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider
  --conf spark.hadoop.fs.s3a.impl.disable.cache=true
  --packages org.apache.hadoop:hadoop-aws:3.2.0,com.amazonaws:aws-java-sdk-bundle:1.11.375
  target/wine-quality-1.0-SNAPSHOT.jar
```
