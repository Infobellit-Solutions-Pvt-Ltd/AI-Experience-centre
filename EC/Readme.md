# AI-Experience Centre
 This is an application where user can interact with AI to get experience with different features.

## Features
- Image to text
- Text to Image
- Retrieval augmented Generation(RAG)
<!-- - Text to video -->

## Requirements
- Docker(to run the containers)
- 16GB GPU(to run the models)

## What user can Experience?
##### Image to text
##### Text to image
##### RAG [Click here for more...](./backend/API/RAG/Readme.md)

## Installation

There are two ways to run the backend server for the Experience centre
1. Using the existing docker images from the repo.
2. Using the Flask files for backend  

```sh
clone the repo to your machine

# To run the existing docker images
# Enter into the directory
cd /EC/

# Run the below script which will run the existing images of UI, depended models and APIs
sh build.sh
```
```sh

clone the repo to your machine

# create a virtual environment
<python -m venv env>
# activate it
<env/scripts/activate>

# Install the requirements into the environment
pip install -r requirements.txt

# Run specific backend API's for specific features
# RAG
python '/backend/API/RAG/request.py' or 
navigate to the './backend/API/RAG/' and follow the setup instructions.

# Run the below script for UI 
EC/frontend/UI-image.sh

# Run the models on the server you had
EC/backend/<Server you used for deployment>

# Example
EC/backend/TGI-M/LLM-RAG.sh
```
**Note:** If you are using the custom models, then replace the deployment code in the /backend/Models and start working on the specific features.