FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ src/
COPY templates/ templates/

# Copy ONLY the ChromaDB database
COPY data/chroma_db/ data/chroma_db/

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["python", "app.py"]