# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python script
COPY palmer-panguin-decisionTree.py .

# Run the script when container starts
CMD ["python", "palmer-panguin-decisionTree.py"]