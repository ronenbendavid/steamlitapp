FROM python:3.7
EXPOSE 8501
RUN mkdir -p /app
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run app.py