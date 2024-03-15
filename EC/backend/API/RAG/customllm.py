import os
from glob import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, JSONLoader, CSVLoader, BSHTMLLoader, Docx2txtLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from rag.Link_scraper import LinkScraper
from rag.Text_extractor import TextExtractor
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import warnings
from flask import Flask, jsonify, request
from flask_cors import CORS
import datetime
import yaml
import requests
import json
import faiss

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------

def index_tracker():
    try:
        # Create an empty dictionary
        empty_data = {}
        file_name = "faiss_index/data.json"

        # Check if the file exists
        if not os.path.exists(file_name):
            # Save the empty dictionary to a JSON file
            with open(file_name, 'w') as file:
                json.dump(empty_data, file)
            print("Empty JSON file created:", file_name)
        else:
            print("File already exists:", file_name)
    except Exception as e:
        print(f"Error in creating the 'faiss_index/data.json':{e}")

index_tracker()

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
        # with open('documents/content.txt', "r",encoding='utf-8') as f:
        #     text_content = f.read()
        # print(text_content)
        load_docs_and_save(directory='documents')
        # return jsonify({"text": text_content})
        # return acknowledgment
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
    # numberOfLinks = int(data.get('numberOfLinks'))/
    numberOfLinks = 1
 
    # Process the extracted link as needed (e.g., store in the database)
    print("Extracted Link:", extracted_link)
    print("Number Of Links: " , numberOfLinks)
    ack = scrape(extracted_link, numberOfLinks)
    print(ack)
    # load_docs_and_save(directory="documents")
    # Return a response if needed
    return jsonify({"message": "Success"})

# <------------------------------------------------------------Data reading part------------------------------------------------------------------->

@app.route('/getData' , methods=['GET'])
def getData():
    with open('documents/Content.txt' , 'r',encoding='utf-8') as f:
        fileContent =  f.read()
    print(fileContent)
 
    return jsonify({'text':fileContent})

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
    doc_file.save(os.path.join('./documents', doc_file.filename))
    load_docs_and_save(directory="documents")
    # print("File saved successfully")
    # return "Blob Success"
    return jsonify({"message": "Success"})

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
data = load_data(file_name)
loaded_docs = set()
# Function to load documents from the directory
def load_docs_and_save(directory):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    documents = []

    for file_path in glob(os.path.join(directory, "*")):
        if file_path in loaded_docs:
            continue
        else:
            loaded_docs.add(file_path)   
            print(f"Converting {file_path} into embeddings")
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
                print("Printing the length of the Chunk : ", totalChunks)
                for doc in loaded_docs:
                    doc.metadata["unique_id"] = unique_id  # Adding unique ID to metadata
                    title = unique_id
                    # print("Metadata of the document:", doc.metadata)  # Print metadata to check if unique_id is added
                documents.extend(loaded_docs)    
            except Exception as e:
                print(f"Error loading document '{file_path}': {e}")
            # print(documents)
        try:
            index = faiss.read_index("faiss_index/index.faiss")
            totalEmbeddings = index.ntotal
        except:
            totalEmbeddings = 0
        data[title]={"chunks":totalChunks , "ntotal":totalEmbeddings}
        print("Printing the New chunk and Existing Embedding :",data)
        # Save the updated data back to the file
        save_data(data, file_name)

        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings)

        db = FAISS.from_documents(documents, embeddings)
        new_db.merge_from(db)

        new_db.save_local("faiss_index")
        print("Embeddings saved successfully")
    # return embeddings
    return "Documents loaded"


 ##### ------------------ > Removing the Available embedding <----------------- ###########


def get_and_remove_chunks(data, index, identifier):
    def get_chunk_range(data, identifier):
        # Extracting the numeric part from the identifier
        ntotal = data[identifier]['ntotal']
        chunks = data[identifier]['chunks']
        start_chunk = ntotal - chunks + 1
        
        # Generating the range
        chunk_range = list(range(start_chunk, ntotal + 1))
        
        return chunk_range
    
    # Get chunk range
    chunk_range = get_chunk_range(data, identifier)
    
    return chunk_range

import numpy as np
input_profile = input("Enter the filename to remove the embeddings")
if input_profile:
    index = faiss.read_index("faiss_index/index.faiss")
    file_name = "faiss_index/data.json"
    data = load_data(file_name)

    print("check embeddings after addings" ,index.ntotal)

    removed_chunks = get_and_remove_chunks(data, index, input_profile)
    print("Removed chunks:", removed_chunks)
    print("length of Removed chunks:", len(removed_chunks))
    # Convert chunk range to numpy array
    ids_to_remove_array = np.array(removed_chunks, dtype=np.int64)

    # Remove IDs from index
    index.remove_ids(ids_to_remove_array)

    # index = faiss.read_index("faiss_index/index.faiss")
    print("check embeddings after Removed the index : " ,index.ntotal)
    faiss.write_index(index, "faiss_index/index.faiss")
else:
    # print("input_profile not received, skipping the execution of the code")
    pass



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
    try:
        data = request.get_json()
        query = data.get('message', '')
        embeddings = get_embeddings()
        db_name = "faiss_index"
        if os.path.exists(db_name):
            new_db = FAISS.load_local(db_name, embeddings,allow_dangerous_deserialization=True)
            print("Searching...")
            retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            relevant_documents = retriever.get_relevant_documents(query)
            # print(relevant_documents)
            passage = []
            meta_data = []
            for doc in relevant_documents:
                sub_passage = doc.page_content
                sub_metadata = doc.metadata
                passage.append(sub_passage)
                meta_data.append(sub_metadata)
            # print("Extracted context",passage,"\nMetadata:" ,meta_data)

            # print(query)
            context = passage
            prompt_template = f"""
            Context:{context}
            ###
            Answer the following question using only information from the Context. 
            Answer in a complete sentence, with proper capitalization and punctuation. 
            If there is no good answer in the Context, say "I don't know".
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
            return reply
        else:
            return 'No database file found'
        # return "Query Recieved"
    except Exception as e:
        print(f"Error occurred during generation: {e}")
        return jsonify({"error": str(e)}), 500
    


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=8888)