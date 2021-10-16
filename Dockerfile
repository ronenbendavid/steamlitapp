FROM python:3.8
RUN mkdir -p /usr/src/steamlitapp
WORKDIR /usr/src/steamlitapp

# Installing requirements
COPY ./steamlitapp/requirements.txt ./steamlitapp/requirements.txt
RUN pip3 install  -r ./steamlitapp/requirements.txt

# Microservices code copy
COPY ./steamlitapp /usr/src/steamlitapp

WORKDIR /usr/src
RUN chmod -R 777 /usr/src/steamlitapp
# For local testing
EXPOSE 8081
CMD ["python", "./steamlitapp/app.py"]
