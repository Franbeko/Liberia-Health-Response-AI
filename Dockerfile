# Use official Python image as base
FROM python:3.11-slim

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file and setup.py
COPY requirements.txt .
COPY setup.py .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (adjust if your app uses a different port)
EXPOSE 8000

# Command to run the app (adjust as needed)
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
