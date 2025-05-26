# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce layer size
# And --default-timeout to prevent issues with slow connections
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the src directory into the container at /app/src
COPY src/ src/

# Copy the data directory into the container at /app/data
# training script will need to access data/raw
COPY data/ data/

# Define the command to run your application
# This script encapsulates the logic from model_training.ipynb
CMD ["python", "src/pipeline/training_pipeline.py"] 