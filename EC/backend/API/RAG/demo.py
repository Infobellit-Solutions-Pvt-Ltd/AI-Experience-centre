from langchain.vectorstores.faiss import FAISS
from request import get_embeddings, get_llm, load_docs_and_save
import os

load_docs_and_save(directory="documents")
def generator():
    try:
        watsonxllm = get_llm()
        print("Accessing model:",watsonxllm.model_id)
        embeddings = get_embeddings()
        print("getting embedding")
        db_name = "faiss_index"
        if os.path.exists(db_name):
            new_db = FAISS.load_local(db_name, embeddings,allow_dangerous_deserialization=True)
            print("Database getting access")
            print("Searching...")
            query = input("Enter the ")
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
            # print(prompt_template)
            response = watsonxllm.generate(prompt=prompt_template)
            print('Response',response['results'][0]['generated_text'])
            return response
        else:
            return 'No database file found'
    except Exception as e:
        print(f"Error occurred during generation: {e}")
        return None
    

generator()