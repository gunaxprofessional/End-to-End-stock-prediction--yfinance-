from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    'owner': 'alpha-quant',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

ARTIFACT_MOUNT = Mount(
    target='/app/artifacts',
    source='stock-artifacts',
    type='volume'
)

with DAG(
    'data_pipeline',
    default_args=default_args,
    description='Data ingestion and feature engineering pipeline',
    schedule='0 5 * * 1-5',  # Run at 5 AM on weekdays (before model training)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data', 'pipeline', 'ingestion', 'features'],
) as dag:

    data_ingestion = DockerOperator(
        task_id='data_ingestion',
        image='stock-data-ingestion:latest',
        api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='stock-network',
        mounts=[ARTIFACT_MOUNT],
        mount_tmp_dir=False,
    )

    feature_engineering = DockerOperator(
        task_id='feature_engineering',
        image='stock-feature-engineering:latest',
        api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='stock-network',
        mounts=[ARTIFACT_MOUNT],
        mount_tmp_dir=False,
    )

    data_ingestion >> feature_engineering
