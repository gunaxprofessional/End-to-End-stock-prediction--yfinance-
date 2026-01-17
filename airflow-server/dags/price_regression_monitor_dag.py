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
    'price_regression_monitor',
    default_args=default_args,
    description='Price regression model monitoring - update actuals and generate drift reports',
    schedule='0 18 * * 1-5',  # Run at 6 PM on weekdays (after market close)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['model', 'price', 'monitoring', 'drift'],
) as dag:

    update_actuals = DockerOperator(
        task_id='update_actuals',
        image='stock-price-trainer:latest',
        api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='stock-network',
        mounts=[ARTIFACT_MOUNT],
        mount_tmp_dir=False,
        command='python -m models.price_regression.update_actuals',
    )

    run_monitor = DockerOperator(
        task_id='run_monitor',
        image='stock-price-trainer:latest',
        api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='stock-network',
        mounts=[ARTIFACT_MOUNT],
        mount_tmp_dir=False,
        command='python -m models.price_regression.monitor',
    )

    update_actuals >> run_monitor
