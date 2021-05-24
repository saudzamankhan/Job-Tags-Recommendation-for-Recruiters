FROM ubuntu:latest

RUN apt-get update -y 
RUN apt-get install python3.7
RUN apt-get install -y python3-pip python3-dev

WORKDIR /

COPY . /

RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD python3 /train_and_serve_model/serve_model.py
#CMD python3 /train_and_serve_model/train_model.py && train_and_serve_model/serve_model.py


#=============================
#ENTRYPOINT [ "python3" ]
#CMD [ "build_model.py" ]


#Docker commands
# For use on command line
# docker build -t nlpproject:latest .
# docker run -it -p 5000:5000 nlpproject

#For Jenkins pipeline script, use the commands below (bat in case of windows and ssh in case of linux)
#bat 'docker build -t nlpproject .''
#bat 'docker run -d -p 5000:5000 nlpproject'