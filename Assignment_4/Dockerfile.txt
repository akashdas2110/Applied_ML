# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app.py score.py /app/

# Install any needed packages specified in requirements.txt
# Ensure you have a requirements.txt file with Flask and other dependencies
COPY requirements.txt /app/
COPY saved_model.sav /app/
COPY template /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which your Flask app runs
EXPOSE 5000
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["python", "app.py"]