FROM python:3.10.12

WORKDIR /t2vapp

COPY 'requirements.txt' 'T2V_api.py'  /t2vapp/

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx


CMD ["python", "T2V_api.py"]




