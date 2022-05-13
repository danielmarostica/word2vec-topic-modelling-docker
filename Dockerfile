FROM python:3.7-slim-buster

COPY requirements.txt requirements.txt

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download pt_core_news_md
RUN pip3 freeze

WORKDIR /app

COPY . .

ENV AWS_DEFAULT_REGION us-east-1

ENTRYPOINT ["python3", "inference.py", "instagram"]
