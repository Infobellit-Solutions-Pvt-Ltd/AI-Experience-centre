### FastAPI Endpoint 
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from transformers import AutoProcessor, AutoModelForVision2Seq

app = FastAPI()

# Load processor and model
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

@app.post("/generate-text")
async def generate_text(file: UploadFile = File(...)):
    try:
        # Open the image from the uploaded file
        image = Image.open(io.BytesIO(await file.read()))

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

        return {"processed_text": processed_text}

    except Exception as e:
        return {"error": str(e)}
