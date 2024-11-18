FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Update pip to the latest version
RUN pip install --upgrade pip

# Install the dependencies
RUN pip install --default-timeout=100 -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Flask will run on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
