FROM ubuntu:latest

RUN apt-get update -y 
RUN apt-get install python3.7
RUN apt-get install -y python3-pip python3-dev

WORKDIR /

COPY . /

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python3" ]

CMD [ "server.py" ]

# docker build -t nlpproject:latest .
# docker run -it -p 5000:5000 nlpproject