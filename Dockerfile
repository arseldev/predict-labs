# Use a lightweight official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (libgomp1 is required by LightGBM/XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directory structure for volumes (data, logs, models)
RUN mkdir -p data logs models && chmod -R 777 data logs models

# Expose port (if you want to add a monitoring API later, default is none)
# EXPOSE 8000

# Set default entrypoint command to run simulation
CMD ["python", "main.py", "--mode", "simulate"]
