# GET the python latest version
FROM python:3.9-slim

# Set the wroking directory in the container
WORKDIR /app

# Copy the dependencies file
COPY requirements.txt .

# install the dependencies
RUN pip install -r requirements.txt

# Install faiss
RUN pip install faiss-cpu

# Copy the flask code app to the container
COPY . .

# Copy the config file
COPY config.yaml .

# Expose the port
EXPOSE 8888

# Specify the commmand to run on container to start
CMD [ "python", "request.py" ]