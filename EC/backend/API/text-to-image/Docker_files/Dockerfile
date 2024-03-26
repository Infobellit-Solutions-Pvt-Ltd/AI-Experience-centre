FROM python

WORKDIR /app

RUN mkdir /app/images

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install Python dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Run load_model
COPY load_model.py ./
RUN python load_model.py

# Copy the code
COPY flask_backend1.py ./

# Expose port for communication with your UI
EXPOSE 8027

# Start the Flask API
CMD ["python", "./flask_backend1.py"] 
