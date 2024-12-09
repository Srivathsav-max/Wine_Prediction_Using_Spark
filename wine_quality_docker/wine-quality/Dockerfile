FROM openjdk:11-jdk-slim

# Set environment variables
ENV SPARK_VERSION=3.2.0
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Install required packages
RUN apt-get update && \
    apt-get install -y curl wget && \
    rm -rf /var/lib/apt/lists/*

# Download and setup Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz

# Create directories for data and models
RUN mkdir -p /app/data/predictions /app/data/test_predictions /app/models

# Set AWS credentials as environment variables
ENV AWS_ACCESS_KEY_ID=your_access_key
ENV AWS_SECRET_ACCESS_KEY=your_secret_key
ENV AWS_REGION=your_region

WORKDIR /app

# Copy the JAR file
COPY target/wine-quality-1.0-SNAPSHOT.jar /app/wine-quality.jar

# Create simple entrypoint script
RUN echo '#!/bin/bash\n\
spark-submit \
    --master local[*] \
    --driver-memory 4g \
    --executor-memory 4g \
    --class com.mlearning.spark.TrainAndPersistWineQualityDataModel \
    /app/wine-quality.jar' > /app/run.sh && \
    chmod +x /app/run.sh

ENTRYPOINT ["/app/run.sh"]

