#!/bin/bash

# Backend images 
"""
RAG, Text to image, image to text
"""
# Function to pull Docker image
pull_image() {
    image_name=$1
    echo "Pulling Docker image: $image_name"
    if docker pull "$image_name"; then
        echo "Successfully pulled $image_name"
    else
        echo "Failed to pull $image_name"
        exit 1
    fi
}

# Function to run Docker container
run_container() {
    image_name=$1
    port=$2
    echo "Running Docker container for $image_name on port $port"
    if docker run -d -p "$port":5000 "$image_name"; then
        echo "Successfully started container for $image_name"
    else
        echo "Failed to start container for $image_name"
        exit 1
    fi
}

# Pull and run the image for kbhavana13/image_to_text
pull_image "kbhavana13/image_to_text:latest"
run_container "kbhavana13/image_to_text:latest" 5000

# Pull and run the image for gyash99/text2img
pull_image "gyash99/text2img"
run_container "gyash99/text2img" 8026

# Pull and run the image for tejavardhanreddy/ragbackend:v1
pull_image "tejavardhanreddy/ragbackend:v1"
run_container "tejavardhanreddy/ragbackend:v1" 8888
