FROM python:3.8
RUN mkdir -p ~/project
WORKDIR /project 
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt --user 
COPY . .
WORKDIR /project/app/
ENTRYPOINT ["python", "/project/app/app.py"] 
EXPOSE 5000