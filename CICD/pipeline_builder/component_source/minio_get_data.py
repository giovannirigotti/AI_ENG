# load dependancies
import kfp as kfp
import kfp.dsl as dsl
from kfp import components
import os
from kfp.components import InputPath, OutputPath, create_component_from_func
from minio import Minio
import urllib3
import datetime as dt
import pandas as pd


def get_data_from_minio(
    minio_path: str,
    bucket: str,
    dest_file_path: OutputPath(),
):

    import numpy
    from io import BytesIO
    import pandas as pd
    import urllib3
    from minio import Minio
    import os
    import pyarrow

    print("WOAH")

    print(os.getenv('MINIO_ACCESS_KEY'))

    client = Minio(
        "storage-api.course.aiengineer.codex-platform.com",
        access_key=os.getenv('MINIO_ACCESS_KEY'),
        secret_key=os.getenv('MINIO_SECRET_KEY'),
        secure=True,
        http_client=urllib3.PoolManager(

            retries=urllib3.Retry(
                total=5,
                backoff_factor=0.2,
                status_forcelist=[500, 502, 503, 504],
            ),
        ),
    )
    buff = client.list_buckets()
    for b in buff:
        print(b.name)
        # Get data from minio using get_object, decode it using BytesIO and read the parquet result with pandas
    try:
        response = client.get_object(bucket, minio_path)
        print(response)
        # Read data from response.
        parquet_object = BytesIO(response.data)
        data = pd.read_parquet(parquet_object)
    finally:
        response.close()
        response.release_conn()
    # pass dataset to component output
    data.to_parquet(dest_file_path)


if __name__ == '__main__':

    create_component_from_func(
        get_data_from_minio,
        output_component_file='components/get_data_from_minio.yaml',
        base_image='python:3.8',
        packages_to_install=[
            'numpy==1.21.6',
            'minio==6.0.2',
            'pandas==1.0.5',
            'pyarrow==10.0.1'
        ],
    )
