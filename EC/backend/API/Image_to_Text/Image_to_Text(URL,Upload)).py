import io
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load processor and model
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

@app.route("/generate-text", methods=["POST"])
def generate_text():
    try:
        data = request.json
        image_url = data.get("url")
        Image_file = request.files.get("file")

        if image_url:
            # If URL is provided, download the image from the URL
            image = Image.open(requests.get(image_url, stream=True).raw)
        elif Image_file:
            # If file is provided, open the image from the uploaded file
            image = Image.open(io.BytesIO(Image_file.read()))
        else:
            return jsonify({"error": "No input provided"}), 400

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
        processed_text = processor.post_process_generation(generated_text)

        return jsonify({"processed_text": processed_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":  
    app.run(debug=True, host='192.168.1.1', port=5000)

