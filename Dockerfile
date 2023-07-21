# Use the official Python image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the required files to the working directory
COPY requirements.txt /app
COPY app.py /app
COPY templates /app/templates
COPY cardio_train.csv /app
COPY Dockerfile /app

RUN pip install --upgrade pip

# Install required Python packages
RUN pip install -r requirements.txt

# Expose the port that the Flask app will run on
EXPOSE 80

# Start the Flask app when the container starts
CMD ["python", "app.py"]
