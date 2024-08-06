import os
import pandas as pd

from datetime import datetime, timedelta

from sklearn.datasets import make_regression

from airflow import DAG
from airflow.decorators import task
from airflow.operators.email import EmailOperator

from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

from utils import create_dummy_dataset


# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 29),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'feature_drift_monitoring',
    default_args=default_args,
    description='A DAG to monitor feature drift weekly',
    schedule_interval='0 0 * * 5',  # Every Friday at midnight
) as dag:

    @task
    def generate_latest_data():
        # Generate latest dummy data
        reference_data = create_dummy_dataset(n_samples=100, n_features=4, noise=0.1, random_state=2)
        return reference_data


    @task
    def compare_data(reference_data):
        # Creating the previous training data
        latest_data = create_dummy_dataset(n_samples=100, n_features=4, noise=0.1, random_state=42)

        def test_drift_detection():
            drift_flag = False

            data_drift = TestSuite(
                tests=[
                    DataDriftTestPreset(stattest_threshold=0.1)
                ]
            )

            data_drift.run(reference_data=reference_data, current_data=latest_data)
            test_summary = data_drift.as_dict()
            feature_result = test_summary["tests"][0]["parameters"]["features"]

            # For storing result
            result = {col: [] for col in feature_result}

            for test in feature_result:
                if feature_result[test]["detected"]:
                    drift_flag = True
                    result[test].append(True)

            drift_report = pd.DataFrame(result, columns=list(feature_result.keys()))
        
            return {'drift_flag': drift_flag, 'drift_report': drift_report}
        
        return test_drift_detection()

    @task
    def send_email_if_drift(drift_data):
        if drift_data["drift_flag"]:
            email_operator = EmailOperator(
                task_id='send_email',
                to=['rojen@fusemachines.com'],
                cc=['shreejan@fusemachines.com'],
                subject='Feature Drift Detected',
                html_content=f"""{drift_data["drift_report"].to_html()}"""
            )

            email_operator.execute(context=None)


    # Define task dependencies
    reference_data = generate_latest_data()
    drift_data = compare_data(reference_data)
    send_email_if_drift(drift_data)