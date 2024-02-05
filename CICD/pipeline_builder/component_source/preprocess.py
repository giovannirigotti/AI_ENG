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


def preprocess_data(
    input_data_path: InputPath('parquet'), 
    preprocess_train_test_data: OutputPath(),
    preprocess_validation_data: OutputPath(),
    label_column: int = 0,
):

    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import IsolationForest
    # Missing values imputations with mean values
    from sklearn.impute import SimpleImputer

    
    ### load data ###
    
    df = pd.read_parquet(
        input_data_path,
    )
    
    ### autoclean data to allow only copatible types in features
    numerics = ['int','float']
    df = df.select_dtypes(include=numerics)

    ### separate train_test from validation

    # Create our imputer to replace missing values with the mean e.g.
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_df = imp.fit(df)

    # Impute our data, then transform train_test dataset 
    df_imp = imp_df.transform(df)

    # Instanciate isolation forest to get rid of outliers
    isolate_forest = IsolationForest(n_jobs=-1, random_state=1)
    isolate_forest.fit(df_imp)
    isolate_predictions =isolate_forest.predict(df_imp)

    ### clean dataset with results
    df_isolated = df_imp[np.where(isolate_predictions == 1, True, False)]
    
    ### back to dataframe
    dataset = pd.DataFrame(df_isolated,columns = df.columns)
    
    ### separate train_test from validation
    train_test,validation_set = train_test_split(dataset, test_size=0.2)
    
    ### write to parquet
    train_test.to_parquet(preprocess_train_test_data)
    validation_set.to_parquet(preprocess_validation_data)

if __name__ == '__main__':


    create_component_from_func(
        preprocess_data,
        output_component_file='components/preprocess_data.yaml',
        base_image='python:3.8',
        packages_to_install=[
            'numpy==1.21.6',
            'xgboost==1.1.1',
            'pandas==1.0.5',
            'tensorboardX==2.5.1',
            'scikit-learn==1.0',
            'pyarrow==10.0.1'
        ],
    )