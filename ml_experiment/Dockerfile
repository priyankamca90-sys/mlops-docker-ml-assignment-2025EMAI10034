FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (so pip can see it)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

CMD ["python", "palmer-panguin-decisionTree.py"]