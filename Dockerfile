FROM python:3.8
COPY . /steamlitapp
WORKDIR /steamlitapp
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]