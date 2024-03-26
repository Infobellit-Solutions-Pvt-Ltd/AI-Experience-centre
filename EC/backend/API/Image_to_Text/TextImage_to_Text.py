from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# # Load image from URL
# url =  'https://img.freepik.com/premium-vector/creative-illustrator-editable-text-effect_179567-292.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

image_path = "quote.png"
image = Image.open(image_path).convert("RGB")

# Load processor
processor = TrOCRProcessor.from_pretrained('facebook/nougat-base')

# Load model
model = VisionEncoderDecoderModel.from_pretrained('facebook/nougat-base')

# Process image with processor
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Generate text with adjusted length
generated_ids = model.generate(pixel_values, max_new_tokens=260)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)  # Print the extracted text

# https://i.postimg.cc/ZKwLg2Gw/367-14.png
