# import depandancies
import kfp as kfp
import kfp.dsl as dsl
from kfp import components
import os
import datetime as dt
from kfp.components import InputPath, OutputPath, create_component_from_func

get_data_from_minio_op = components.load_component_from_file('components/get_data_from_minio.yaml')
xgboost_predict_on_csv_op=components.load_component_from_file('components/xgb_predict.yaml')
xgboost_train_on_csv_op=components.load_component_from_file('components/xgb_train_dbg.yaml')
preprocess_data_op =components.load_component_from_file('components/preprocess_data.yaml')
save_xgboost_model_bst_op = components.load_component_from_file('components/save_xgboost_model_bst.yaml')


user='guillaume-etevenard'
namespace = f'kubeflow-user-{user}'

dsl.pipeline(name='xgboost_chicago_base')
def xgboost_pipeline_upgraded(namespace=namespace):
    import datetime
    from kfp.onprem import use_k8s_secret
    
    bucket='guillaume-etevenard'
    
    data = get_data_from_minio_op(
        minio_path = 'datasets/chicago/trips.parquet',
        bucket = bucket,
    )
    
    data.apply(
        use_k8s_secret(
            secret_name='minio-service-account',
            k8s_secret_key_to_env={
                'access_key':'MINIO_ACCESS_KEY',
                'secret_key':'MINIO_SECRET_KEY'
            }
        )
    )
    
    preprocessed_data = preprocess_data_op(
        input_data = data.output
    ).set_memory_limit('1Gi')
    
    
    # Training and prediction on dataset in CSV format
    model_trained_on_csv = xgboost_train_on_csv_op(
        training_data=preprocessed_data.outputs['preprocess_train_test_data'],
        label_column=0,
        objective='reg:squarederror',
        num_iterations=200,
    ).set_memory_limit('1Gi').outputs
    
    xgboost_predict_on_csv_op(
        data=preprocessed_data.outputs['preprocess_validation_data'],
        model=model_trained_on_csv['model'],
        label_column=0,
    ).set_memory_limit('1Gi')
    
    saved = save_xgboost_model_bst_op(
        bucket=bucket,
        input_model = model_trained_on_csv['model'],
        minio_model_path ='models/frompipeline/xgboost/chicago'
    ).set_memory_limit('1Gi')
    
    saved.apply(
        use_k8s_secret(
            secret_name='minio-service-account',
            k8s_secret_key_to_env={
                'access_key':'MINIO_ACCESS_KEY',
                'secret_key':'MINIO_SECRET_KEY'
            }
        )
    )



### a token has been automatically provided in the KF_PIPELINES_SA_TOKEN_PATH variable. This token allow accès to only your namespace
token_file = os.getenv("KF_PIPELINES_SA_TOKEN_PATH")
with open(token_file) as f:
    token = f.readline()
client = kfp.Client(host='http://ml-pipeline.kubeflow.svc.cluster.local:8888',
               existing_token=token)

### create the experiment
EXPERIMENT_NAME = 'Aiengineer labs session5'
experiment = client.create_experiment(name=EXPERIMENT_NAME, namespace=namespace)

### compile the pipeline
kfp.compiler.Compiler().compile(
    pipeline_func=xgboost_pipeline_upgraded,
    package_path='xgboost_pipeline_upgraded.yaml')

### submit the base pipeline
client.create_recurring_run(
    experiment_id=experiment.id,
    job_name = "XGB_chicago_upgraded",
    cron_expression = '30 16 * * *',
    pipeline_package_path  = 'xgboost_pipeline_upgraded.yaml', 
)


