FROM python:3.8-buster

# Code is copied on build. In development, we mount a local folder over a copied one.
COPY ../requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends build-essential gcc libsndfile1 

RUN pip3 install -r requirements.txt
