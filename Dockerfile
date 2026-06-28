# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies first (cached layer)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (yield_model.pkl is excluded via .dockerignore)
COPY . /app

RUN chmod +x /app/start.sh

EXPOSE 7860

ENTRYPOINT ["/app/start.sh"]
