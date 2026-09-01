FROM python:3.10-slim

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose standard Cloud Run port (8080)
EXPOSE 8080

# Run FastAPI server on port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
