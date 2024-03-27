import io
import os
import datetime
import requests
import base64
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load processor and model
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

@app.route("/generate-text", methods=["POST"])
def generate_text():
   try:
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs('images', exist_ok=True)
    if 'file' in request.files:
        img_file = request.files['file']
        img_file_path = os.path.join('images', 'image.png')
        img_file.save(img_file_path)
        image = Image.open(img_file_path)
        print("Image saved successfully")

    elif 'url' in request.form:
        url  = request.form.get('url')
        image = Image.open(requests.get(url, stream=True).raw)
        img_file_path = 'images/image_' + timestamp + '.jpg'
        image.save(img_file_path)
        print("Image downloaded and saved successfully")

    else:
        return "Please provide a valid URL for image download or upload a valid image file."

    with open(img_file_path, "rb") as file:
        img_data = file.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")

    image = Image.open(io.BytesIO(img_data))
    print("Image saved in Bytes successfully ")

    # Process the image and generate text
    inputs = processor(text="<grounding>An image of", images=image, return_tensors="pt")
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # By default, the generated text is cleaned up.
    processed_text = processor.post_process_generation(generated_text)[0]
    print("processed_text:" , processed_text)

    return {
    "message": processed_text,
    "image_base64": img_base64
   }

   except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='192.168.0.231', port=5000)
