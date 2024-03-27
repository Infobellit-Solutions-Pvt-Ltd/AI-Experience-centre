
import streamlit as st
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import os
import base64
import tempfile
import subprocess

# Load the DiffusionPipeline model
pipe = DiffusionPipeline.from_pretrained("ali-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

st.title("Text to Video Streamlit App")

# Get user input for the prompt
prompt = st.text_area("Enter your prompt here:")

# Video options
st.sidebar.header("Video options")
num_frames = st.sidebar.slider("Number of frames", min_value=16, max_value=384, value=64)
fps = st.sidebar.slider("Frames per second", min_value=5, max_value=30, value=6)
resolution = st.sidebar.selectbox("Resolution", options=["320x240", "640x480", "1280x720"])

if st.button("Generate Video"):
    # Generate video
    video_frames = pipe(prompt, num_inference_steps=25, num_frames=num_frames).frames[0]

    # Export video
    video_path = "video.mp4"  # Save the video in the same directory
    export_to_video(video_frames, output_video_path=video_path, fps=fps)

    # Convert video to H264 codec
    h264_video_path = "video_h264.mp4"
    subprocess.run(["ffmpeg", "-i", video_path, "-vcodec", "libx264", h264_video_path])

    # Display video
    st.video(h264_video_path)

    # Optionally clean up temporary directory after some time
    # time.sleep(60)  # Delete directory after 60 seconds (adjust as needed)
    # shutil.rmtree(temp_dir)

st.write("Note: This app requires a CUDA GPU for faster processing.")

