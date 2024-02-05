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



def xgboost_predict(
    data_path: InputPath('CSV'),  # Also supports LibSVM
    model_path: InputPath('XGBoostModel'),
    predictions_path: OutputPath('Predictions'),
    label_column: int = None,
):
    '''Make predictions using a trained XGBoost model.
    Args:
        data_path: Path for the feature data in CSV format.
        model_path: Path for the trained model in binary XGBoost format.
        predictions_path: Output path for the predictions.
        label_column: Column containing the label data.
    '''
    from pathlib import Path

    import numpy
    import pandas
    import xgboost
    import pyarrow

    df = pandas.read_parquet(
        data_path,
    )
    
    ### autoclean data to allow only copatible types in features
    numerics = ['int','float']
    df = df.select_dtypes(include=numerics)

    if label_column is not None:
        df = df.drop(columns=[df.columns[label_column]])

    testing_data = xgboost.DMatrix(
        data=df,
    )

    model = xgboost.Booster(model_file=model_path)

    predictions = model.predict(testing_data)

    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    numpy.savetxt(predictions_path, predictions)

if __name__ == '__main__':


    create_component_from_func(
        xgboost_predict,
        output_component_file='components/xgb_predict.yaml',
        base_image='python:3.7',
        packages_to_install=[
            'xgboost==1.1.1',
            'pandas==1.0.5',
            'pyarrow==10.0.1'
        ],
    )