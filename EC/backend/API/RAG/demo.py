# from app import get_llm
# import os

# wxa_api_key = "S9JiP4YV24yiXG1H6m3ZVkzWQ8VcVxVRMg61nDxOx1ps" # IBM APIKEY
# wxa_project_id = "eed30038-0353-4e6a-b2b4-22f2fcd17161" # Watsonx projectID
# wxa_url = "https://eu-de.ml.cloud.ibm.com"  #Franfurt
# os.environ['WATSONX_APIKEY'] = wxa_api_key
# os.environ['WATSONX_PROJECT_ID'] = wxa_project_id
# os.environ['WATSONX_URL'] = wxa_url
# model = get_llm(wxa_api_key,wxa_project_id,wxa_url)

# print(model.generate(prompt = "Who is the founder of IBM"))
import requests

url = "https://tgi-deployment.1dxnn8ccpevd.eu-de.codeengine.appdomain.cloud/generate"
data = {
    "inputs": "What is DeepLearning?",
    "parameters": {
        "max_new_tokens": 20
    }
}
headers = {
    "Content-Type": "application/json"
}
print("Searching...")
response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print(response.json())
else:
    print("Failed to send request")
