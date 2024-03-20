import os,textwrap
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# It reads all the text available in the website and store in a text file 
class TextExtractor:
    def __init__(self,max_depth = 2, path_to_save_links="./Extracted_links.txt", path_to_save_text="./documents/Content.txt"):
        self.max_depth = max_depth
        self.path_to_save_links = path_to_save_links
        self.path_to_save_text = path_to_save_text 

    # RecursiveUrlLoader is a method to load all URLs under a root directory and is provided by langchain
    # BeautifulSoupTransformer is a method for parsing HTML content
    # Head_Url extractor
    def Text_Extractor(self, url):
        try:
            loader = RecursiveUrlLoader(url=url, max_depth=self.max_depth)
            bs_transformer = BeautifulSoupTransformer()
            docs = bs_transformer.transform_documents(loader.load(), tags_to_extract=["p"])
            return docs
        except Exception as e:
            print(f"Error occurred Text_Extractor : {e}")
            return None

    # The links are loaded from a text file
    def Load_links(self):
        try:
            with open(self.path_to_save_links, "r", encoding="utf-8") as file:
                links = [link.strip() for link in file.readlines()]
            return links
        except Exception as e:
            print(f"Error occurred in Load_Links: {e}")
            return None
        
    # Child_URL extractor 
    # The text extraction and saved to a specified file
    def Extract_and_save_text(self, links):
        try:
            file_path = self.path_to_save_text

            if os.path.exists(file_path):
                os.remove(file_path)
            
            if not os.path.exists("documents"):
                os.makedirs("documents")
            

            for link in links:
                docs = self.Text_Extractor(link)
                with open(self.path_to_save_text, "+a", encoding="utf-8") as f:
                    for doc in docs:
                        wrapped_content = textwrap.fill(doc.page_content, width=100)  # Change 80 to your desired width
                        f.write(wrapped_content + "\n")
                        # f.write(doc.page_content + "\n")

                return docs
        except Exception as e:
            print(f"Error occured in Extract_and_save_text: {e}")
            return None

    # This method makes to be automate all the calls for the above methods  
    def Extract_text(self):
        try:
            links = self.Load_links()
            docs = self.Extract_and_save_text(links)
            return docs
        except Exception as e:
            print(f"Error occured in Extract_text: {e}")
            return None
    