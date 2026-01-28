from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_DIR = "D:/Programas/Codes for VS code/Proyecto_F1Kedro/proyecto_f1kedro"

with DAG(
    dag_id="kedro_f1_training",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    run_training = DockerOperator(
        task_id="run_training",
        image="proyecto_f1kedro-jupyter:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        command="bash -lc 'cd /workspace && kedro run --pipeline=training'",
        mounts=[Mount(source=PROJECT_DIR, target="/workspace", type="bind")],
        network_mode="proyecto_f1kedro_default",
    )
