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
    print(data)
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
    print("File saved successfully")
    return "Blob Success"

# <------------------------------------------Loading all the documents from a directory and embedding them-----------------------------------------> 

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
                    print(loader)
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    print(loader)
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
                elif file_path.endswith(('.docx', '.doc')):
                    loader = Docx2txtLoader(file_path)
                    print(loader)
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
                elif file_path.endswith(('.html', '.htm')):
                    loader = BSHTMLLoader(file_path)
                    print(loader)
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
                elif file_path.endswith('.json'):
                    loader = JSONLoader(file_path)
                    print(loader)
                    documents.extend(loader.load_and_split(text_splitter=text_splitter))
            except Exception as e:
                print(f"Error loading document '{file_path}': {e}")
        embeddings = get_embeddings()
        db = FAISS.from_documents(documents, embeddings)
        db.save_local("faiss_index")
        print("Embeddings saved successfully")
    # return embeddings
    return "Documents loaded"

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

# <----------------------------------------------------------Getting the WatsonxLLM---------------------------------------------------------------->

def get_llm():
    """
    Create a WatsonxLLM model with the specified parameters.
    
    Returns:
        Model: A WatsonxLLM model object.
        
    Raises:
        Exception: If there is an error creating the WatsonxLLM model.
    
    wxa_api_key = "S9JiP4YV24yiXG1H6m3ZVkzWQ8VcVxVRMg61nDxOx1ps" # IBM APIKEY
    wxa_project_id = "eed30038-0353-4e6a-b2b4-22f2fcd17161" # Watsonx projectID
    wxa_url = "https://eu-de.ml.cloud.ibm.com"  #Franfurt
    """
    wxa_api_key = os.getenv("WATSONX_APIKEY")
    wxa_project_id = os.getenv("WATSONX_PROJECT_ID")
    wxa_url = os.getenv("WATSONX_URL")
    print(wxa_api_key,wxa_project_id,wxa_url)
    try:
        parameters = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 100
        }

        model = Model(
            model_id='ibm/granite-13b-chat-v2',
            params=parameters,
            credentials={
                "url": wxa_url,
                "apikey": wxa_api_key,
            },
            project_id=wxa_project_id
        )
        return model

    except Exception as e:
        raise Exception("Error creating WatsonxLLM model: {}".format(e))

# <---------------------------------------------------------------Response Generator--------------------------------------------------------------->

@app.route("/generator",methods = ["POST","GET"])
def generator():
    try:
        data = request.get_json()
        query = data.get('message', '')
        watsonxllm = get_llm()
        embeddings = get_embeddings()
        print('LLM',watsonxllm.model_id)
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
            print("Extracted context",passage,"\nMetadata:" ,meta_data)

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
            print(prompt_template)
            response = watsonxllm.generate(prompt=prompt_template)
            answer = response['results'][0]['generated_text']
            # print(answer)
            return answer
        else:
            return 'No database file found'
    except Exception as e:
        print(f"Error occurred during generation: {e}")
        return None

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=8888)