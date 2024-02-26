import re
import requests
from bs4 import BeautifulSoup

class LinkScraper:
    # Reads all the inputs provided by the user
    def __init__(self, head_url, max_length, num_of_urls, path_to_save_links = "Extracted_links.txt"):
        self.head_url = head_url
        self.max_length = max_length
        self.num_of_urls = num_of_urls
        self.path_to_save_links = path_to_save_links
        self.all_urls = []

    # Links Extraction from the website and returns a list of URLs
    def Extract_links(self, url):
        try:
            response = requests.get(url)
            page = BeautifulSoup(response.content, "html.parser")
            urls = []
            count = 0
            for tag in page.find_all("a"):
                href = tag.get("href")
                if href and re.match(r"^https?://", href) and count < self.num_of_urls:
                    urls.append(href)
                    count += 1
            return urls
        except Exception as e:
            print(f"Error in Extract_links: {e}")
            return None
        
    # Recursive extraction of links in hierarchical manner (see the architecture)
    def Extract_child_links(self, head_urls):
        try:
            child_urls = []
            for link in head_urls:
                if len(self.all_urls + child_urls) < self.max_length:
                    urls = self.Extract_links(link)
                    child_urls.extend(urls)
                else:
                    self.all_urls.extend(child_urls)
                    return
            self.all_urls.extend(child_urls)
            if child_urls:
                self.Extract_child_links(child_urls)
        except Exception as e:
            print(f"Error in Extract_child_links: {e}")
            return None

    # Save the links to specified file or provided path
    def Save_extracted_links(self):
        try:
            max_urls = self.all_urls[:self.max_length]
            with open(self.path_to_save_links, "w", encoding="utf-8") as f:
                for url in max_urls:
                    f.write(url + "\n")
        except Exception as e:
            print(f"Error in save_extracted_links: {e}")
            return None
        
    # Function to call all the methods internally
    def Links_Extractor(self):
        try:
            self.all_urls.append(self.head_url)
            head_urls = self.Extract_links(self.head_url)
            self.Extract_child_links(head_urls)
            self.Save_extracted_links()
        except Exception as e:
            print(f"Error occured in Links_Extractor : {e}")
            return None