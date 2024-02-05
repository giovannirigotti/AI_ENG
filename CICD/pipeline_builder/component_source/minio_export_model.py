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

def save_xgboost_model_bst(
    bucket: str,
    input_model_path: InputPath('XGBoostModel'),
    minio_model_path:str
):
    '''Make predictions using a trained XGBoost model.
    Args:
        bucket: Bucket name used in Minio to store the model 
        model_path: Path for the trained model in binary XGBoost format.
    '''
    import xgboost
    import urllib3
    from minio import Minio
    from datetime import datetime
    import os

    # load model using input model_path
    model = xgboost.Booster(model_file=input_model_path)

    model_dir = "."
    BST_FILE = "model.bst"
    model_file = os.path.join((model_dir), BST_FILE)
    model.save_model(model_file)

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

    ### define path where the object will be stored
    minio_model_name = f'{minio_model_path}/{BST_FILE}'
    ### put object
    client.fput_object(bucket, minio_model_name, BST_FILE)

if __name__ == '__main__':

    create_component_from_func(
        save_xgboost_model_bst,
        output_component_file='components/save_xgboost_model_bst.yaml',
        base_image='python:3.8',
        packages_to_install=[
            'minio==6.0.2',
            'xgboost==1.1.1',
        ],
    )