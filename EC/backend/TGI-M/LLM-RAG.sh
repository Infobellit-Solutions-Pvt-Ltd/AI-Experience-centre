"""Write a script to automate the deployment of all models"""

# model = stabilityai/stablelm-2-zephyr-1_6b
# model = meta-llama/Llama-2-7b-chat-hf
model = google/flan-t5-base
volume = $PWD/data # share a volume with the Docker container to avoid downloading weights every run

token = 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

#docker run --cpus 16 --name cpu-model-flan-t5-base -d --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --dtype bfloat16
docker run --gpus all --name gpu-model-all-gpus-1 -d --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8084:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id $model --dtype bfloat16