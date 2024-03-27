import streamlit as st
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import os
import io
import base64
import tempfile
import subprocess
from flask import Flask, request, jsonify ,send_file
from flask_cors import CORS


# Load the DiffusionPipeline model
pipe = DiffusionPipeline.from_pretrained("ali-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

app = Flask(__name__)
CORS(app)

@app.route("/generate_video", methods=["POST"])
def generate_video():
    data = request.get_json()
    prompt = data["prompt"]
    print(prompt)
    num_frames = data.get("num_frames", 64)
    fps = data.get("fps", 6)

    # Generate video
    video_frames = pipe(prompt, num_inference_steps=25, num_frames=num_frames).frames[0]

    # Export video
    video_path = "video.mp4"  # Save the video in the same directory
    export_to_video(video_frames, output_video_path=video_path, fps=fps)

    # Convert video to H264 codec
    h264_video_path = "video_h264.mp4"
    subprocess.run(["ffmpeg", "-i", video_path, "-vcodec", "libx264", h264_video_path])

    # Read the video file as bytes
    with open(h264_video_path, "rb") as file:
        video_bytes = file.read()

    return send_file(
        io.BytesIO(video_bytes),
        mimetype='video/mp4',
        as_attachment=True,
        download_name="test.mp4"
    )

if __name__ == "__main__":
    app.run(debug=True, host='192.168.0.139', port=9090)

