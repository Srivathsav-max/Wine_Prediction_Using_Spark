# Wine Quality Prediction Using Parallel Computing on AWS

## Overview
This project demonstrates how to train a Wine Quality Prediction ML model using Apache Spark in a parallel computing environment with multiple EC2 instances. The training leverages Spark's MLlib for distributed model training and uses AWS S3 for data storage. Additionally, this setup uses Ansible to automate the provisioning and configuration of a Spark cluster across 4 EC2 instances (1 master and 3 workers). After training, the model can be packaged and run inside a Docker container to simplify deployment.

## Prerequisites
- AWS EC2 instances (4 total: 1 master, 3 workers)
- AWS S3 bucket for storing datasets and model outputs
- AWS Access Key and Secret Key for S3 access
- Ansible installed on your local machine
- Maven installed on the master node or locally if building before deployment
- Docker 

## Steps

### 1. Clone the Repository
Clone the repository :
```bash
git clone https://github.com/Srivathsav-max/Wine_Prediction_Using_Spark.git
```
You can do this locally or directly on the AWS master instance. For convenience, you can SCP (secure copy) the repository or simply re-clone it inside the AWS master instance after it's provisioned.

<img width="1908" alt="Screenshot 2024-12-08 at 11 25 51 PM" src="https://github.com/user-attachments/assets/35a5f6e1-3ce1-4e06-820a-331ff9993a4e">


### 2. Set Up the Spark Cluster with Ansible
Ansible is used to set up the Spark environment across all 4 instances (1 master + 3 workers).

- Ensure your `inventory.ini` file contains the IP addresses or hostnames of your EC2 instances, and that you've set up SSH access (via `.pem` key) for Ansible.
- Run the setup playbook:
  ```bash
  ansible-playbook -i inventory.ini setup.yml
  ```
  
This will install all required packages, set up Java, Spark, and any dependencies, and start the Spark master and worker services across the instances.

#### Stopping the Spark Cluster
To stop the Spark services across instances:
```bash
ansible-playbook -i inventory.ini stop-spark.yml
```

#### Starting the Spark Cluster
If you stopped the cluster and want to start it again:
```bash
ansible-playbook -i inventory.ini start-spark.yml
```

### 3. Accessing the Master Node
Once the Ansible setup is done, you can SSH into the master node using its IP and your `.pem` file:
```bash
ssh -i yourkey.pem ubuntu@<master-node-ip>
```
From the master node, you can manage the Spark cluster, run training jobs, and monitor the Spark UI (if ports are open and accessible).

### 4. Prepare Your Data on S3
Before running the code, upload your training, validation, and test datasets to an S3 bucket. For example:

- `TrainingDataset.csv`
- `ValidationDataset.csv`

<img width="961" alt="Screenshot 2024-12-08 at 11 26 47 PM" src="https://github.com/user-attachments/assets/054be94d-3912-48fd-bb8b-d05d3e792fae">


Replace these paths in the code if needed to point to your actual S3 locations. Also, update the code with your AWS credentials and S3 paths in the main files so the model can be saved and loaded correctly.

### 5. Building the Project
On the master node (or locally, then transfer the files):
```bash
cd wine_quality/wine-quality
mvn clean package -DskipTests
```
This will resolve all dependencies and produce the JAR file (e.g., `target/wine-quality-1.0-SNAPSHOT.jar`).

### 6. Running the Training Job
Run the Spark training job from the master node. Make sure the Spark cluster is running and accessible at `spark://<master-node-ip>:7077`. Use your AWS credentials and S3 paths:

```bash
spark-submit \
  --class com.mlearning.spark.TrainAndPersistWineQualityDataModel \
  --master spark://<master-node-ip>:7077 \
  --conf spark.executor.memory=3g \
  --conf spark.driver.memory=3g \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.hadoop.fs.s3a.access.key=acess_key \
  --conf spark.hadoop.fs.s3a.secret.key=secret_key \
  --conf spark.hadoop.fs.s3a.path.style.access=true \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider \
  --conf spark.hadoop.fs.s3a.impl.disable.cache=true \
  --packages org.apache.hadoop:hadoop-aws:3.2.0,com.amazonaws:aws-java-sdk-bundle:1.11.375 \
  target/wine-quality-1.0-SNAPSHOT.jar
```

### Output:
<img width="1087" alt="Screenshot 2024-12-08 at 10 05 25 PM" src="https://github.com/user-attachments/assets/ef6a6bb1-bbea-46f6-b2a6-d60aa1f8b7e6">
<img width="1081" alt="Screenshot 2024-12-08 at 11 20 59 PM" src="https://github.com/user-attachments/assets/c85ea3f1-0bff-47a2-850e-9357acd4863a">


### Spark Session:
<img width="1909" alt="Screenshot 2024-12-08 at 11 18 56 PM" src="https://github.com/user-attachments/assets/5a7986d4-8281-444e-b1c1-81d351b37ab3">


This will:
- Pull the training and validation data from S3.
- Train the model using logistic regression distributed across 4 EC2 instances.
- Validate and tune the model.
- Save the trained model back to the S3 bucket.

### 7. Running in Docker
A Dockerfile is included to run the same training and prediction processes inside a Docker container. This simplifies deployment and makes it easier to run the prediction application on a single EC2 instance without installing Spark locally.

or

You can use my public Docker Image
```
https://hub.docker.com/r/srivathsav03/wine_quality_prediction/tags
```
<img width="1907" alt="Screenshot 2024-12-08 at 11 20 12 PM" src="https://github.com/user-attachments/assets/a26a9610-422f-4398-aefd-61cae83ae481">


**Building the Docker Image:**
```bash
docker build -t wine-quality .
```

**Running the Container:**
```bash
docker run -e AWS_ACCESS_KEY_ID=access_key   -e AWS_SECRET_ACCESS_KEY=secret_key   -e AWS_REGION=us-east-1   -v $(pwd)/data:/app/data   -v $(pwd)/models:/app/models   wine-quality
```

<img width="1083" alt="Screenshot 2024-12-09 at 1 32 28 AM" src="https://github.com/user-attachments/assets/0aa56dc6-6191-4e94-a6e9-92c5da7ba2b6">


Make sure you have created `spark-network` and your Spark master is accessible within that network. Also ensure `HOME` or ivy cache directory issues are resolved as discussed.

## Conclusion
Following the steps above:
- Use Ansible to provision and control a Spark cluster on 4 EC2 instances.
- Upload data to S3 and run Spark jobs to train and validate the ML model in parallel.
- Finally, run the prediction or training process inside a Docker container for easier deployment and environment consistency.

By adhering to these steps, you have a fully automated, reproducible environment for parallel ML model training on AWS with Spark, and a Dockerized setup for easy portability.
