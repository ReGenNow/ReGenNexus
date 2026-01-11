FROM python:3.9-slim

LABEL maintainer="ReGen Designs LLC"
LABEL description="ReGenNexus Core - Universal Agent Protocol"
LABEL version="0.1.1"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Expose the default port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV REGENNEXUS_SECURITY_LEVEL=2

# Create a non-root user to run the application
RUN useradd -m regennexus
USER regennexus

# Command to run the application
CMD ["regennexus-registry"]
