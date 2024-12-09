FROM bitnami/spark:3.4.0

USER root
WORKDIR /app

# Copy your JAR file
COPY target/wine-quality-1.0-SNAPSHOT.jar /app/wine-quality.jar

# Create a 'spark' user with a home directory
RUN useradd -ms /bin/bash spark && \
    chown -R spark:spark /app

# Set HOME and USER environment variables
ENV HOME=/home/spark
ENV USER=spark

USER spark
ENTRYPOINT ["spark-submit"]
