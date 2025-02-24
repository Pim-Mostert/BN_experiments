#!/bin/bash

# Define the .env file
ENV_FILE=".env"

# Check if the .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE file not found!"
    exit 1
fi

# Read AZURE_ARTIFACTS_TOKEN from .env file
AZURE_ARTIFACTS_TOKEN=$(grep '^AZURE_ARTIFACTS_TOKEN=' "$ENV_FILE" | cut -d'=' -f2-)

if [ -z "$AZURE_ARTIFACTS_TOKEN" ]; then
    echo "Error: AZURE_ARTIFACTS_TOKEN is not set in $ENV_FILE!"
    exit 1
fi

# Build the Docker image
docker build --build-arg AZURE_ARTIFACTS_TOKEN=$AZURE_ARTIFACTS_TOKEN -t sometag .
