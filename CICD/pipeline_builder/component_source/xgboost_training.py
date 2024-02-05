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


def xgboost_train(
    training_data_path: InputPath('CSV'),  # Also supports LibSVM
    model_path: OutputPath('XGBoostModel'),
    model_config_path: OutputPath('XGBoostModelConfig'),
    training_log_path: OutputPath(),
    starting_model_path: InputPath('XGBoostModel') = None,


    label_column: int = 0,
    num_iterations: int = 10,
    booster_params: dict = None,

    # Booster parameters
    objective: str = 'reg:squarederror',
    booster: str = 'gbtree',
    learning_rate: float = 0.3,
    min_split_loss: float = 0,
    max_depth: int = 6,
):
    '''Train an XGBoost model.
    Args:
        training_data_path: Path for the training data in CSV format.
        model_path: Output path for the trained model in binary XGBoost format.
        model_config_path: Output path for the internal parameter configuration of Booster as a JSON string.
        starting_model_path: Path for the existing trained model to start from.
        label_column: Column containing the label data.
        num_boost_rounds: Number of boosting iterations.
        booster_params: Parameters for the booster. See https://xgboost.readthedocs.io/en/latest/parameter.html
        objective: The learning task and the corresponding learning objective.
            See https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
            The most common values are:
            "reg:squarederror" - Regression with squared loss (default).
            "reg:logistic" - Logistic regression.
            "binary:logistic" - Logistic regression for binary classification, output probability.
            "binary:logitraw" - Logistic regression for binary classification, output score before logistic transformation
            "rank:pairwise" - Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized
            "rank:ndcg" - Use LambdaMART to perform list-wise ranking where Normalized Discounted Cumulative Gain (NDCG) is maximized

    '''
    import pandas
    import xgboost
    from sklearn.metrics import roc_curve
    from tensorboardX import SummaryWriter
    from sklearn.model_selection import train_test_split
    import pyarrow

    ### embedded function to allow tensorboard to monitor the training ###

    def TensorBoardCallback():
        writer = SummaryWriter(training_log_path)

        def callback(env):
            for k, v in env.evaluation_result_list:
                print(k, v)
                writer.add_scalar(k, v, env.iteration)

        return callback

    ### load data ###

    df = pandas.read_parquet(
        training_data_path,
    )

    # autoclean data to allow only copatible types in features
    numerics = ['int', 'float']
    df = df.select_dtypes(include=numerics)

    ### split data ###

    data = df.drop(columns=[df.columns[label_column]])
    label = df[df.columns[label_column]]
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.2, random_state=100)

    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)

    ### model HP ###

    booster_params = booster_params or {}
    booster_params.setdefault('objective', objective)
    booster_params.setdefault('booster', booster)
    booster_params.setdefault('learning_rate', learning_rate)
    booster_params.setdefault('min_split_loss', min_split_loss)
    booster_params.setdefault('max_depth', max_depth)

    ### Not from scratch training management ###

    starting_model = None
    if starting_model_path:
        starting_model = xgboost.Booster(model_file=starting_model_path)

    ### Model fit to data ###

    model = xgboost.train(
        params=booster_params,
        dtrain=dtrain,
        num_boost_round=num_iterations,
        xgb_model=starting_model,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        callbacks=[TensorBoardCallback()]
    )

    ### Save the model as an artifact ###
    model.save_model(model_path)

    model_config_str = model.save_config()
    with open(model_config_path, 'w') as model_config_file:
        model_config_file.write(model_config_str)


if __name__ == '__main__':

    create_component_from_func(
        xgboost_train,
        output_component_file='components/xgb_train_dbg.yaml',
        base_image='python:3.7',
        packages_to_install=[
            'xgboost==1.1.1',
            'pandas==1.0.5',
            'tensorboardX==2.5.1',
            'scikit-learn==1.0',
            'pyarrow==10.0.1'
        ],
    )
