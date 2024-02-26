# GET the python latest version
FROM python:3.9-slim

# Set the wroking directory in the container
WORKDIR /app

# Copy the dependencies file
COPY requirements.txt .

# Update pip and setuptools
RUN pip install --upgrade pip setuptools

# install the dependencies
RUN pip install -r requirements.txt

# Copy the flask code app to the container
COPY . .

# Expose the port
EXPOSE 8888

# Specify the commmand to run on container to start
CMD [ "python", "request.py" ]