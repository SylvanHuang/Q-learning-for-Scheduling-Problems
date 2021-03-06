FROM python:3.8-slim-buster

WORKDIR /app

COPY src .
RUN pip3 install -r requirements.txt
CMD python3 $PROBLEM
