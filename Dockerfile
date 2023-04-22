FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app
RUN apt-get update -y && apt-get install awscli -y
RUN apt-get update && pip3 install -r requirements.txt
CMD ["python", "app.py"]