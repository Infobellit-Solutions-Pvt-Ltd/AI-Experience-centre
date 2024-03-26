from flask import Flask, request, jsonify
from diffusers import DiffusionPipeline
import torch
import io
from flask_cors import CORS
from PIL import Image
import base64

app = Flask(__name__)
CORS(app)

# Load the model from the specified path (replace with your actual path)
model_path = "/home/yashwanthgundala6/project/text-to-image"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_gpu=True)
pipe.to("cuda")

@app.route("/generate_image", methods=["POST"])
def generate_image():
  try:
    data = request.get_json()
    print(data)
    prompt = data.get('prompt')
    print(prompt)

    with torch.cuda.device("cuda"):
      images = pipe(prompt=prompt).images[0]

    image_path = "/app/images/generated_image.png"
    images.save(image_path)

    image_bytes = open(image_path, 'rb').read()
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    print(f"Image saved successfully at: {image_path}")

    return {
      "message": prompt,
      "image_base64": img_base64
    }

  except Exception as e:
    print(f"Error: {e}")
    return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
  app.run(host="0.0.0.0", debug=True, port=8027, use_reloader=False)

