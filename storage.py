import os
import boto3
import pandas as pd
from io import StringIO, BytesIO
from botocore.exceptions import NoCredentialsError

from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET_NAME

class MinioArtifactStore:
    def __init__(self, bucket_name=BUCKET_NAME):
        self.bucket_name = bucket_name
        self.s3_endpoint = MINIO_ENDPOINT
        self.access_key = MINIO_ACCESS_KEY
        self.secret_key = MINIO_SECRET_KEY
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except:
            print(f"Bucket {self.bucket_name} does not exist. Creating...")
            self.s3_client.create_bucket(Bucket=self.bucket_name)

    def upload_file(self, local_path, s3_path):
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
            print(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_path}")
        except FileNotFoundError:
            print(f"The file {local_path} was not found")
        except NoCredentialsError:
            print("Credentials not available")

    def download_file(self, s3_path, local_path):
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_path, local_path)
            print(f"Downloaded s3://{self.bucket_name}/{s3_path} to {local_path}")
        except Exception as e:
            print(f"Error downloading {s3_path}: {e}")

    def save_df(self, df, s3_path):
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_path,
            Body=csv_buffer.getvalue()
        )
        print(f"DataFrame saved to s3://{self.bucket_name}/{s3_path}")

    def load_df(self, s3_path):
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_path)
            df = pd.read_csv(BytesIO(obj['Body'].read()))
            print(f"DataFrame loaded from s3://{self.bucket_name}/{s3_path}")
            return df
        except Exception as e:
            print(f"Error loading DataFrame from {s3_path}: {e}")
            raise
