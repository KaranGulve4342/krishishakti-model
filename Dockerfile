# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install curl + hf CLI (needed for HF Bucket downloads)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://hf.co/cli/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies first (cached layer)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (yield_model.pkl is excluded via .dockerignore)
COPY . /app

RUN chmod +x /app/start.sh

EXPOSE 7860

ENTRYPOINT ["/app/start.sh"]
