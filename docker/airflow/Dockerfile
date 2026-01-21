FROM apache/airflow:2.10.3-python3.11

ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-2.10.3/constraints-3.11.txt"

COPY requirements-airflow.txt /tmp/requirements-airflow.txt

RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt --constraint "${CONSTRAINT_URL}"
