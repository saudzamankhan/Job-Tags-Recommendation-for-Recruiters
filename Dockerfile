FROM ubuntu:latest

RUN apt-get update -y 
RUN apt-get install python3.7
RUN apt-get install -y python3-pip python3-dev

WORKDIR /

COPY . /

RUN pip3 install -r requirements.txt

EXPOSE 5001

ENTRYPOINT [ "python3" ]
CMD [ "serve_model.py" ]

#=============================
#CMD python3 serve_model.py
#CMD python3 train_model.py && serve_model.py
#=============================

#Docker commands
# For use on command line
# docker build -t nlpproject:latest .
# docker run -it -p 5000:5000 nlpproject

#For Jenkins pipeline script, use the commands below (bat in case of windows and ssh in case of linux)
#bat 'docker build -t nlpproject .''
#bat 'docker run -d -p 5000:5000 nlpproject'