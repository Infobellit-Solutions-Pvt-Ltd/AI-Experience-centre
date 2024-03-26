### flask Image upload ###

import io
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from flask import Flask , request, send_file, jsonify
from flask_cors import CORS
import os
import base64
 
 
app = Flask(__name__)
CORS(app)
 
# Load processor and model
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
 
# Image to Text (Bhavana)
@app.route("/uploadImg", methods=['POST'])
def uploadImg():
    try:
        os.makedirs('images', exist_ok=True)
        if 'file' not in request.files:
           return jsonify({"error": "No file provided"}), 400
        img_file = request.files['file']
        if img_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        img_file.save(os.path.join('images', 'img.png'))
        print("File saved successfully")
 
        image_path="./images/img.png"
        print(image_path)
 
        # Open the image and convert to bytes
        with open(image_path, "rb") as file:
                img_data = file.read()
                img_base64 = base64.b64encode(img_data).decode("utf-8")
 
        #print(image_bytes)
        # Open the image from bytes using PIL
        image = Image.open(io.BytesIO(img_data))
        print("Image saved successfully")
 
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
 
        # By default, the generated text is cleaned up, and the entities are extracted.
        processed_text = processor.post_process_generation(generated_text)[0]

        print("processed_text" , processed_text)

        return {
        "message": processed_text,
        "image_base64": img_base64
    }
 
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
if __name__ == "__main__":
    app.run(debug=True, host='192.168.0.231', port=7000)
