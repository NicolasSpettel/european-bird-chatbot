# Use a lightweight Python base image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flask and other required libs
RUN pip install flask

# Create the directory structure needed for your code and data
RUN mkdir -p src/agents src/database src/tools
RUN mkdir -p data/chroma_db


# Copy the core agent file and its direct dependencies
COPY .env .
COPY app.py .
COPY src/agents/bird_qa_agent.py src/agents/
COPY src/database/chroma_client.py src/database/
COPY src/config.py src/
COPY src/tools/audio_processor.py src/tools/
COPY src/ src/

# Copy templates
COPY templates/ templates/

# Copy the ChromaDB data folder and its contents
COPY data/chroma_db/ data/chroma_db/

# Expose the port the app runs on
EXPOSE 5000

# Set the command to run the Flask web server
CMD ["python", "app.py"]