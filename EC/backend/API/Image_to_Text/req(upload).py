# Inference with image upload
import requests

# Give the actual URL where your FlaskAPI application is running
api_url ="http://192.168.1.1:5000"

# Path to the image file you want to send for inference
image_path = "Images/Rapunzel.jpg"
                              
# Open the image file
with open(image_path, "rb") as file:
    files = {"file": ("image.jpg", file, "image/jpeg")}
    response = requests.post(api_url, files=files)

# Check the response from the FastAPI endpoint
if response.status_code == 200:
    result = response.json()
    processed_text = result["processed_text"]
    print(processed_text)
else:
    print("Error:", response.text)

