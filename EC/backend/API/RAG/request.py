
import os
from glob import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, JSONLoader, CSVLoader, BSHTMLLoader, Docx2txtLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from rag.Link_scraper import LinkScraper
from rag.Text_extractor import TextExtractor
import requests
import warnings
from flask import Flask, jsonify, request
from flask_cors import CORS
import datetime
import yaml

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

 
# Web Scraping ()
def scrape(head_url,max_length):
    try:
        web_scraper = LinkScraper(head_url, max_length, num_of_urls=1)
        web_scraper.Links_Extractor()
        print("All the available links are saved...")
 
        # Text extraction
        TextExtractor().Extract_text()
 
        print("All the text got scraped")
        acknowledgment = "Web Scraping done..."
        load_docs_and_save(directory='documents')
        return acknowledgment
    except Exception as e:
        print(f"Error occurres in the main function: {e}")
        return None
 


# Webscraping --> get link from front end -> 
@app.route('/extract_link', methods=['POST'])
def extract_link():
    data = request.get_json()
    extracted_link = data.get('link')
    print("Extracted_link",extracted_link)
    numberOfLinks = int(data.get('numberOfLinks'))
 
    # Process the extracted link as needed (e.g., store in the database)
    print("Extracted Link:", extracted_link)
    print("Number Of Links: " , numberOfLinks)
    ack = scrape(extracted_link, numberOfLinks)
    print(ack)
    # load_docs_and_save(directory="documents")
    # Return a response if needed
    return jsonify({"message": "Link received successfully"})

@app.route("/uploadDoc", methods=['POST'])
def uploadDoc():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs('documents', exist_ok=True)
    if 'file' not in request.files:
        return "No file provided", 400
    doc_file = request.files['file']
    if doc_file.filename == '':
        return "No selected file", 400
    doc_file.save(os.path.join('documents', doc_file.filename))
    load_docs_and_save(directory="documents")
    print("File saved successfully")
    return "Blob Success"

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
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
                elif file_path.endswith(('.docx', '.doc')):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
                elif file_path.endswith(('.html', '.htm')):
                    loader = BSHTMLLoader(file_path)
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
                elif file_path.endswith('.json'):
                    loader = JSONLoader(file_path)
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
            except Exception as e:
                print(f"Error loading document '{file_path}': {e}")
        embeddings = get_embeddings()
        db = FAISS.from_documents(documents, embeddings)
        db.save_local("faiss_index")
        print("Embeddings saved successfully")
    # return embeddings
    return "Documents loaded"

def load_config():
    """
    Load configuration from the 'config.yaml' file.
 
    Returns:
        dict: Configuration settings.
    """
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

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



@app.route("/generator",methods = ["POST","GET"])
def generator():
    try:
        data = requests.get_json()
        query = data.get('message', '')
        embeddings = get_embeddings()
        db_name = "faiss_index"
        if os.path.exists(db_name):
                
            new_db = FAISS.load_local(db_name, embeddings)
            print("Searching...")
            retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            relevant_documents = retriever.get_relevant_documents(query)
            print(relevant_documents)
            passage = []
            meta_data = []
            for doc in relevant_documents:
                sub_passage = doc.page_content
                sub_metadata = doc.metadata
                passage.append(sub_passage)
                meta_data.append(sub_metadata)
                # print(sub_passage, sub_metadata)

            print(query)
            prompt = f"""
                Answer the following question by considering the context.
                Question: {query}
                Context: {passage}
                Answer: 
                    """
            url = "http://192.168.0.231:8084/generate"
            data = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 200}
            }
            response = requests.post(url=url, json=data)
            ans = response.json()["generated_text"]
            print(ans)
            return ans
        else:
            return 'No database file found'
    except Exception as e:
        print(f"Error occurred during generation: {e}")
        return None

if __name__=="__main__":
    app.run(debug=True,host = "0.0.0.0",port=8888)