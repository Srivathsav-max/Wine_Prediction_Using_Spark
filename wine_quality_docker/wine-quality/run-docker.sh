
#!/bin/bash

# Build the Docker image
docker build -t wine-quality .

# Run the container with AWS credentials
docker run \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION=$AWS_REGION \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  wine-quality