# Use official Python lightweight image
FROM python:3.9-windowsservercore-1809

# Set working directory inside container
WORKDIR /app

# Copy requirements file (create this next)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]