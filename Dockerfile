# Use a standard Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies (good for any C++ stuff)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project into the container
# This includes main.py, the 'dist' folder, and the 'data' folder
COPY . .

# Tell Docker the app will run on port 7860
EXPOSE 7860

# The command to run your app
# We use gunicorn (a production server) to run the 'app' variable inside the 'main.py' file
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "main:app"]
