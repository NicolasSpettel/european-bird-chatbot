# Use a lightweight Python base image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code and configuration
COPY .env .
COPY app.py .
COPY src/ src/
COPY templates/ templates/

# Create the directory where the data volume will be mounted
RUN mkdir -p /app/data/chroma_db

# Expose the port the app runs on
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=app.py

# Set the command to run the Flask web server
CMD ["python", "app.py"]