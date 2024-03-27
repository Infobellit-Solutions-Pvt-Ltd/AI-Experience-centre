# inference with the URL of the image.
import requests
from PIL import Image

# URL of the Flask API endpoint
api_url = "http://192.168.1.1:5000/generate-text"

# Provide either a URL or an image file
data = {
    "url": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png",
}

# Send POST request to the API endpoint
response = requests.post(api_url, json=data)

 # Alternatively, you can provide an image file:
# Path to the image file you want to send for inference
# file = "Input/Rapunzel.jpg"

# # Open the image file
# with open(file, "rb") as file:
#     files = {"file": ("image.jpg", file, "image/jpeg")}
#     response = requests.post(api_url, files=files)

# Send POST request to the API endpoint
# response = requests.post(url, files=files)

# Check if the request was successful
if response.status_code == 200:
    result = response.json()
    processed_text = result.get("processed_text")
    if processed_text:
        print(processed_text)
    else:
        print("No processed text returned")
else:
    print("Error:", response.text)
