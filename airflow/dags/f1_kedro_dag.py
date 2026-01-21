from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# IMPORTANTE: en Windows, usa ruta con / (no \) para que Docker Desktop la interprete bien
PROJECT_DIR = "D:/Programas/Codes for VS code/Proyecto_F1Kedro/proyecto_f1kedro"

with DAG(
    dag_id="kedro_f1_pipelines",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    run_classification = DockerOperator(
        task_id="run_classification",
        image="proyecto_f1kedro-jupyter:latest",  # imagen que SI tiene kedro
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        command="bash -lc 'cd /workspace && kedro run --pipeline=classification'",
        mounts=[
            Mount(source=PROJECT_DIR, target="/workspace", type="bind"),
        ],
        # Si necesitas conectarte a postgres/otros servicios por nombre,
        # usa la misma red de compose:
        network_mode="proyecto_f1kedro_default",
    )

    run_regression = DockerOperator(
        task_id="run_regression",
        image="proyecto_f1kedro-jupyter:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        command="bash -lc 'cd /workspace && kedro run --pipeline=regression'",
        mounts=[
            Mount(source=PROJECT_DIR, target="/workspace", type="bind"),
        ],
        network_mode="proyecto_f1kedro_default",
    )

    run_classification >> run_regression
