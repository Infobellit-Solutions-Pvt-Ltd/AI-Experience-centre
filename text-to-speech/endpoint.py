from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from transformers import AutoProcessor, AutoModel
from pydantic import BaseModel
import tempfile
import uuid
import os
import torch
from scipy.io.wavfile import write
from typing import Tuple

app = FastAPI()

MODEL_NAME = 'suno/bark'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME)
model = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
model.to(DEVICE)


class AudioRequest(BaseModel):
    text: str
    preset: str

def generate_audio(text: str, preset: str) -> Tuple[str, str]:
    print(text)
    inputs = processor(text, voice_preset=preset, return_tensors='pt').to(DEVICE)
    audio_array = model.generate(**inputs, do_sample=True)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.config.sample_rate
    file_id = uuid.uuid1()
    file_path = os.path.join(
        tempfile.gettempdir(),
        f'{file_id}.wav'
    )
    write(file_path, rate=sample_rate, data=audio_array)
    return file_path, f"{file_id}.wav"

#creating the endpoint 
@app.post("/text_to_speech/generate_audio")
async def generate_audio_endpoint(request: AudioRequest):
    try:
        file_path, filename = generate_audio(request.text, request.preset)
        return StreamingResponse(open(file_path, "rb"), media_type="audio/wav", headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

