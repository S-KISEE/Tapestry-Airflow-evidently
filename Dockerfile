FROM apache/airflow:2.9.1

COPY requirements.txt .

RUN pip install apache-airflow==2.9.1 -r requirements.txt
