import requests

def test_extract_link():
    url = 'http://localhost:8888/extract_link'  # Replace with your actual URL
    data = {
        'link': 'https://cloud.ibm.com/docs/containers?topic=containers-getting-started',
        'numberOfLinks': '5'
    }
    response = requests.post(url, json=data)
    print(response.status_code == 200)
    print(response)

if __name__ == '__main__':
    test_extract_link()
