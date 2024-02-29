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
    - Retrieve relevant docs of the user question using 'similarity_search'
- Foundational model
    - Any foundational model for generating the response from the user questiona with retrieved documents.

## Web Scraping and Text Extraction

The './rag' Python script defines a Flask web application for web scraping and text extraction, using the `LinkScraper` and `TextExtractor` classes from the `rag` package.

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

## Data loading Functions

- `scrape(head_url, max_urls)`: Scrapes a web page given the URL `head_url` and a maximum length `max_urls`.
- `extract_link()`: Extracts a link from the front end and initiates the scraping process.
- `uploadDoc()`: Uploads a document and saves it to a specified directory.

## Document Loading and Processing

- `load_docs_and_save(directory)`: Loads documents from a specified directory and saves them for further processing.
- `get_embeddings()`: Retrieves the embeddings model for the loaded documents.

## Generation

- `generator()`: Generates a response to a given query based on the loaded documents and their embeddings.

## Flask Application

- Defines routes for the web application, such as `/extract_link`, `/uploadDoc`, and `/generator`.
- Uses CORS (Cross-Origin Resource Sharing) for handling requests from different origins.
- Runs the Flask application on `localhost` at port `8888` in debug mode.

Please note that the actual functionality of the code depends on the implementation details of the classes and functions from the `rag` and `langchain` packages.

## Installation
 - create a virtual environment(python -m venv env) pip install -r requirements.txt
