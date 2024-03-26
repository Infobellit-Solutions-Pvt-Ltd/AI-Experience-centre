import requests

url = "http://192.168.1.1:5000/uploadImg"


# Path to the image file you want to send for inference
image_path = "Images/Rapunzel.jpg"

# Open the image file
with open(image_path, "rb") as file:
    files = {"file": ("image.jpg", file, "image/jpeg")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    processed_text = result["processed_text"]
    print(processed_text)
else:
    print("Error:", response.text)