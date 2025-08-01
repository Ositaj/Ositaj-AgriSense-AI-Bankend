# Use Python base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy all files into the container
COPY . .

# Install required packages
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
