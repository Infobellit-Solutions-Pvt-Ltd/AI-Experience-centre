#!/bin/bash

# Arguments: directory path, script name
run_containers() {
    directory=$1
    script=$2

    echo "Running script in $directory"
    if [ -f "$directory/$script" ]; then
        sh "$directory/$script"
    else
        echo "File not found: $directory/$script"
    fi
}

# Run APIs
run_containers "./backend/API" "API-images.sh"

# Run models
run_containers "./backend/TGI-M" "LLM-RAG.sh"

# Run UI
run_containers "./frontend" "UI-image.sh"
