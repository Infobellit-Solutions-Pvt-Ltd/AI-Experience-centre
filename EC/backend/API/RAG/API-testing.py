import requests
import json
from langchain.prompts import PromptTemplate


url = 'http://localhost:8888'  # Change the URL if your server is running on a different address

def test_upload_doc():
    endpooint = f'{url}/uploadDoc'  # Change the URL if your server is running on a different address

    files = {'file': open(f'C:/Users/User/Downloads/Advance_Rag.pdf', 'rb')}  # Replace 'path_to_your_file_here.pdf' with the actual path to your file
    print(files)
    response = requests.post(url=endpooint, files=files)

    print(response.json())

def test_remove_profile():
    endpoint = f"{url}/remove_profile"
    headers = {'Content-Type': 'application/json'}
    data = {
        'profile': 'Advance_Rag.pdf'
    }
    response = requests.post(url=endpoint,headers=headers,json=data)
    print(response.json())
def test_delete_file():
    endpoint = 'http://192.168.0.255:8888/delete_files'
    headers = {'Content-Type':'application/json'}
    data = {
        'file_name':'Teja_CV.pdf'
    }
    response = requests.post(url=endpoint,headers=headers,json=data)
    print(response.json())
def generate_text():
    url = "http://192.168.0.139:8081/generate"
    data = {
        "inputs": 'What is Deep Learning?',
        "parameters": {"max_new_tokens": 500}
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=data)
    # reply = response.json()['generated_text']
    print(type(response.text))

def test_generator():
    # url = 'https://192.168.0.255:8888'
    endpoint = f"{url}/generator"
    # headers = {"Content-Type": "application/json"}
    data={
        'message':input('Enter the Query:')
    }
    response = requests.post(url=endpoint,json=data)
    print(response.text)


def llm():
    query = "Kindly give bullet point summary on the following Content.IBM entered the microcomputer market in the 1980s with the IBM Personal Computer, which soon became known as PC, one of IBM's best selling products. Due to a lack of foresight by IBM,[13][14] the PC was not well protected by intellectual property laws. As a consequence, IBM quickly began losing its market dominance to emerging competitors in the PC market, while at the same time the openness of the PC platform has ensured PC's longevity as the most popular microcomputer standard.Beginning in the 1990s, the company began downsizing its operations and divesting from commodity production, most notably selling its personal computer division to the Lenovo Group in 2005. IBM has since concentrated on computer services, software, supercomputers, and scientific research. Since 2000, its supercomputers have consistently ranked among the most powerful in the world, and in 2001 it became the first company to generate more than 3,000 patents in one year, beating this record in 2008 with over 4,000 patents.[12] As of 2022, the company held 150,000 patents.[15]"
    url = "http://192.168.0.139:8081/generate"
    data = {
        "inputs":query,
        "parameters":{
            "max_new_tokens":250
        }
    }
    response = requests.post(url,json=data)
    # print(query)
    # print(response)
    # return response
    # return {"Success":response.text}
    return type(response)


def test_llm():
    end_point = "http://192.168.0.255:8888/llm"
    # code_output_parser = "Generate the code in proper format with proper indendation"
    # summary_output_parser = "Generate the output as bullet points"
    Content = "These output parsers extract tool calls from OpenAI’s function calling API responses. This means they are only usable with models that support function calling, and specifically the latest tools and tool_choice parameters. We recommend familiarizing yourself with function calling before reading this guide."
    # prompt = f"Summarize the below content and display as bullet points .\n {Content}"
    

    data = {
        # Summarization
        # 'text': f"Summarize the below content and display as bullet points .\n {Content}",
        # Code generation
        # 'text' : "Write a C function to efficiently calculate the Fibonacci sequence. Comment the code liberally to explain what each piece does and why it's written that way"  
        'text' : 'Write a sample code in java to check whether the number 17 is prime or not?'
        # 'text':'What is Deep Learning?'
    }
    
    response = requests.post(url=end_point,json=data)
    print(response.json()[150:-5])
    
    # print(type(dict(response.text)))


def test_scrape():
    # end_point = "http://192.168.0.255:8888/extract_link"
    end_point = "http://192.168.0.255:8888/link"
    data = {
        'link':'https://en.wikipedia.org/wiki/Talk:IBM#Founders'
    }
    response = requests.post(url=end_point,json=data)
    print(response.text)

test_upload_doc()
# test_generator()
# generate_text()
# test_remove_profile()
# test_generator()
# test_llm()
# test_scrape()
