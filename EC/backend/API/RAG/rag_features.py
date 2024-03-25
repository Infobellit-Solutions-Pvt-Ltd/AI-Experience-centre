"""
This code deals with RAG which involves with few features.
1)Web scrapping and Douments uploading
2)Adding and Deleting the embeddings with repsect to documents
3)Generation model deployed in the local server with TGI image
Also interaction with llm 
"""
import os
from glob import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, JSONLoader, CSVLoader, BSHTMLLoader, Docx2txtLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from Scrapping.Link_scraper import LinkScraper
from Scrapping.Text_extractor import TextExtractor
import warnings
from flask import Flask, jsonify, request
from flask_cors import CORS
import datetime
import yaml
import requests
import json
import faiss
import numpy as np

warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app)

# <----------------------------------------------------------Web Scrapping------------------------------------------------------------------------->
 
# Web Scraping ()
def scrape(head_url,max_length):
    try:
        web_scraper = LinkScraper(head_url, max_length, num_of_urls=1)
        web_scraper.Links_Extractor()
        print("All the available links are saved...")
 
        # Text extraction
        TextExtractor().Extract_text()
 
        print("All the text got scraped")
        load_docs_and_save(directory='documents')
    except Exception as e:
        print(f"Error occurres in the main function: {e}")
        return None
 


# Webscraping --> get link from front end -> 
@app.route('/extract_link', methods=['POST'])
def extract_link():
    data = request.get_json()
    print(data)
    extracted_link = data.get('link')
    print("Extracted_link",extracted_link)
    # numberOfLinks = int(data.get('numberOfLinks'))
    numberOfLinks = 1
    print("Extracted Link:", extracted_link)
    print("Number Of Links: " , numberOfLinks)
    ack = scrape(extracted_link, numberOfLinks)
    print(ack)
    load_docs_and_save(directory="documents")
    # Return a response if needed
    return jsonify({"message": "Webscrapping Success","status":200})

# <------------------------------------------------------------Data reading part------------------------------------------------------------------->

@app.route('/getData' , methods=['GET'])
def getData():  
    try:
        with open('documents/Content.txt' , 'r',encoding='utf-8') as f:
            fileContent =  f.read()
        print(type(fileContent))
    
        return jsonify({'text':fileContent,"status":200})
    except FileExistsError as e:
        return jsonify({'text':'Unable to get data',"status":404})
# <----------------------------------------------------Uploading documents from local machine------------------------------------------------------> 

@app.route("/uploadDoc", methods=['POST'])
def uploadDoc():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs('./documents', exist_ok=True)
    if 'file' not in request.files:
        return "No file provided", 400
    doc_file = request.files['file']
    if doc_file.filename == '':
        return "No selected file", 400

    # Check if the file already exists
    file_path = os.path.join('./documents', doc_file.filename)
    if os.path.exists(file_path):
        print(f"File {doc_file.filename} already exists")
    else:
        doc_file.save(file_path)
        load_docs_and_save(directory="documents")
        return jsonify({"message": f"File {doc_file.filename} uploaded"})

    return jsonify({"message": f"File {doc_file.filename} not uploaded (already exists)"})


# <------------------------------------------Loading all the documents from a directory and embedding them-----------------------------------------> 

# Function to load data from a JSON file
def load_data(file_name):
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
# Function to save data to a JSON file
def save_data(data, file_name):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

# File name to store the data
file_name = "faiss_index/data.json"
# data = load_data(file_name)

# Function to load documents from the directory
def load_docs_and_save(directory):

    """
    Load documents from the specified directory, split them into chunks, and save embeddings to a Faiss index.

    Args:
        directory (str): Path to the directory containing the documents.

    Returns:
        str: Status message indicating success or failure.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    documents = []

    for file_path in glob(os.path.join(directory, "*")):  
        # print(f"Converting {file_path} into embeddings")
        try:
            if file_path.endswith(('.txt', '.csv')):
                loader = TextLoader(file_path, encoding="utf-8") if file_path.endswith(".txt") else CSVLoader(file_path, encoding="utf-8")
            elif file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(('.docx', '.doc')):
                loader = Docx2txtLoader(file_path)  
            elif file_path.endswith(('.html', '.htm')):
                loader = BSHTMLLoader(file_path)
            elif file_path.endswith('.json'):
                loader = JSONLoader(file_path)
            unique_id = os.path.splitext(os.path.basename(file_path))[0]
            loaded_docs = loader.load_and_split(text_splitter=text_splitter)
            totalChunks = len(loaded_docs)
            # print("Printing the length of the Chunk : ", totalChunks)
            for doc in loaded_docs:
                doc.metadata["unique_id"] = unique_id  # Adding unique ID to metadata
                title = unique_id
            documents.extend(loaded_docs)    
        except Exception as e:
            print(f"Error loading document '{file_path}': {e}")
        # print(documents)
    embeddings = get_embeddings()
    try:
        index = faiss.read_index("faiss_index/index.faiss")
        totalEmbeddings = index.ntotal
    except:
        totalEmbeddings = 0
        # Create an empty dictionary
        empty_data = {}
        # Directory where the JSON file will be saved
        directory = "faiss_index"
        # Ensure the directory exists, if not, create it
        os.makedirs(directory, exist_ok=True)
        # File name for the empty JSON file
        file_name = os.path.join(directory, "data.json")
        # Save the empty dictionary to a JSON file
        with open(file_name, 'w') as file:
            json.dump(empty_data, file)
        print("Empty JSON file created:", file_name)

    # Load existing data from file or create an empty dictionary if the file doesn't exist
    data = load_data("faiss_index/data.json")
    data[title]={"chunks":totalChunks , "ntotal":totalEmbeddings}
    # print("Printing the New chunk and Existing Embedding :",data)
    # Save the updated data back to the file
    save_data(data, "faiss_index/data.json")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
        db = FAISS.from_documents(documents, embeddings)
        new_db.merge_from(db)
        print("Documents embedded successfully")
        new_db.save_local("faiss_index")
        
    except:
        db = FAISS.from_documents(documents, embeddings)
        db.save_local("faiss_index")
        print("Documents embedded successfully")


 # ------------------------------------------------------ Removing the Available embeddings ------------------------------------------------------->


def get_and_remove_chunks(data, index, identifier):
    try:
        ntotal = data[identifier]['ntotal']
        chunks = data[identifier]['chunks']
        start_chunk = ntotal - chunks + 1
        chunk_range = list(range(start_chunk, ntotal + 1))
        return chunk_range
    except Exception as e:
        print(f"Error in get_and_remove_chunks:{e}")

# ------------------------------------------------- Removing the Available files and embeddings --------------------------------------------------->


def delete_file(file_name):
    try:
        directory = os.path.join("documents",file_name)
        if os.path.exists(directory):
            os.remove(directory)
            return jsonify({"message":f'file {file_name} removed successfully',"status":200})
        else:
            return jsonify({"message":f"File  {file_name} not found","status":404})
    except Exception as e:
        return jsonify({"message":"Unable to delete the file","status":404})

@app.route('/remove_profile', methods=['POST'])
def remove_profile():
    try:
        data = request.get_json()
        if not data or 'profile' not in data:
            return jsonify({"error": "Invalid request. 'profile' key missing in JSON data.","status":400})

        input_profile = data['profile']
        delete_file(file_name=input_profile)

        input_profile_name = input_profile.split(".")[0]
        # print(input_profile_name)

        if input_profile_name:
            index = faiss.read_index("faiss_index/index.faiss")
            file_name = "faiss_index/data.json"
            data = load_data(file_name)
            removed_chunks = get_and_remove_chunks(data, index, input_profile_name)
            ids_to_remove_array = np.array(removed_chunks, dtype=np.int64)
            index.remove_ids(ids_to_remove_array)
            faiss.write_index(index, "faiss_index/index.faiss")

            if input_profile_name in data:
                del data[input_profile_name]
                print(f"Data with {input_profile_name} removed")
            else:
                print(f"Data regarding {input_profile_name} not found")

            save_data(data, file_name)
            data = load_data(file_name)
            # print("Embeddings after removing the index:", index.ntotal)
            return jsonify({"message": f"Profile '{input_profile_name}' removed successfully.","status":200})
        else:
            return jsonify({"message": "Profile name not received.","status":404})
    except FileNotFoundError:
        return jsonify({"error": "File not found.","status":404})
    except Exception as e:
        return jsonify({"error": str(e),"status":404})


# <----------------------------------------------------loading the required embedding model-------------------------------------------------------->

def load_config():
    """
    Load configuration from the 'config.yaml' file.
 
    Returns:
        dict: Configuration settings.
    """
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error in loading the config file: {e}")
config = load_config()
 
def get_embeddings(model_name=config["embeddings"]["name"],
                    model_kwargs={'device': config["embeddings"]["device"]}):
    """
    Load HuggingFace embeddings.
 
    Args:
        model_name (str): The name of the HuggingFace model.
        model_kwargs (dict): Keyword arguments for the model.
 
    Returns:
        HuggingFaceEmbeddings: Embeddings model.
    """
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# <---------------------------------------------------------------Response Generator--------------------------------------------------------------->

@app.route("/generator",methods = ["POST","GET"])
def generator():
    """
    Generator gives the response to the user from the requested question
    based on the documents uploaded/scrapped
    """
    try:
        data = request.get_json()
        query = data.get('message', '')
        print(query)
        embeddings = get_embeddings()
        db_name = "faiss_index"
        if os.path.exists(db_name):
            new_db = FAISS.load_local(db_name, embeddings)
            print("Searching...")
            retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            relevant_documents = retriever.get_relevant_documents(query)
            passage = []
            meta_data = []
            for doc in relevant_documents:
                sub_passage = doc.page_content
                sub_metadata = doc.metadata
                passage.append(sub_passage)
                meta_data.append(sub_metadata)
            context = passage
            prompt_template = f"""
            Context:{context}
            ###
            Instruction : Give the response from the given context in generative approach if data not there from context then say as 'I don't know' and don't make any anwer.
            Obey the Instruction that's an order for you
            Question:{query}
            Answer: 
            """
            url = "http://192.168.0.139:8081/generate"
            data = {
                "inputs": prompt_template,
                "parameters": {"max_new_tokens": 250}
            }
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, headers=headers, json=data)
            reply = response.json()
            ans = str(reply['generated_text'].replace(' n',''))
            print(ans)
            return jsonify({"answer":ans })
        else:
            return jsonify( 'No database file found')
    except Exception as e:
        return jsonify({"Error occurred during generation": str(e)}), 500
    
@app.route('/llm', methods=['POST'])
def llm():
    data = request.get_json()
    query = data['text']
    url = "http://192.168.0.139:8081/generate"
    data = {
        "inputs":query,
        "parameters":{
            "max_new_tokens":250
        }
    }
    response = requests.post(url,json=data)
    ans=response.json()
    return ans['generated_text']


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=8888)
