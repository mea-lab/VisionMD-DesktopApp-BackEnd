# Use the Python base image based on Debian Bookworm Slim
FROM python:3.10-slim-bookworm

# Set the working directory
WORKDIR /app

# Install necessary system dependencies and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the container and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port (optional but good practice)
EXPOSE 8000

# Start the Uvicorn server
CMD ["uvicorn", "backend.asgi:application", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]




