# RAG application
Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response.

## Architecture
- Document Loading
    - WebScraping
    - Uploading documents 
- Embedding documents
    - Huggingface embedding model 
- Storing
    - Save into vector stores
- Retriever
    - Retrieve relevant docs of the user question using 'similarity_search' algorithm
- LLM
    - Any foundational model for generating the response from the user question with retrieved documents.

## Dependencies

- `os`: Provides functions for interacting with the operating system.
- `glob`: Offers a Unix-style pathname pattern expansion.
- `langchain`: A library for text processing and vectorization.
- `ibm_watson_machine_learning`: Accessing Watsonx foundational model.
- `HuggingFace` : Accessing the Embedding models.
- `rag`: A package for web scraping and text extraction.
- `bs4` : Formatting the HTML document
- `requests`: A library for making HTTP requests.
- `warnings`: Allows for handling warning messages.
- `flask & CORS`: Allows to create end-point

## Get Started

The './rag' Python script defines a Flask web application for web scraping(extracts URLs) and text extraction, using the `LinkScraper` and `TextExtractor` classes from the `rag` package.

###  Input Documents
- Web scrapping

    - `/extract_link'`: Scrapes all the available URLs withe the limit of `max_urls` from the given URL `head_url` and save the extracted content in the documents/content.txt.
- Upload local documents
    - `/uploadDoc`:Upload the textual documents from the local machine (supported:.txt,.pdf,.docx,.csv,.json,.html).

### Document Chunking and Embedding

- `load_docs_and_save(directory)`: This method loads documents from a specified directory and saves them using `FAISS` vector db in the `faiss_index` directory for further processing.
- `get_embeddings()`: Gives the embeddings for the loaded documents based on the model from the config file.

### Generation

- `/generator()`: Generates a response to the user question by the relevant data from the loaded documents.

### Other APIs and Flask application
- `/getData`: This API displays the scrapped content to the user
- Uses CORS (Cross-Origin Resource Sharing) for handling requests from different origins.
- Runs the Flask application on `localhost` at port `8888` in debug mode.

<!-- Please note that the actual functionality of the code depends on the implementation details of the classes and functions from the `rag` and `langchain` packages. -->

## Get Started
 - create any virtual environment(python -m venv env) and activate it
 - install the requirements `pip install -r requirements.txt`
 - run the `python app.py` file and the server gets activated with the port `8888`.
