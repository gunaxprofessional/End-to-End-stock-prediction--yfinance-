from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.external_task import ExternalTaskSensor
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
    'price_regression_model',
    default_args=default_args,
    description='Price regression model training pipeline',
    schedule='0 6 * * 1-5',  # Run at 6 AM on weekdays (after data pipeline)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['model', 'price', 'regression', 'training'],
) as dag:

    # Wait for data pipeline to complete
    # execution_delta: data_pipeline runs at 5 AM, this DAG at 6 AM (1 hour difference)
    wait_for_data = ExternalTaskSensor(
        task_id='wait_for_data_pipeline',
        external_dag_id='data_pipeline',
        external_task_id='feature_engineering',
        execution_delta=timedelta(hours=1),
        mode='reschedule',
        timeout=3600,
        poke_interval=60,
    )

    train_model = DockerOperator(
        task_id='train_price_model',
        image='stock-price-trainer:latest',
        api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='stock-network',
        mounts=[ARTIFACT_MOUNT],
        mount_tmp_dir=False,
    )

    wait_for_data >> train_model
