# import depandancies
from kubernetes import client
from kserve import KServeClient
from kserve import constants
from kserve import utils
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1SKLearnSpec
from kserve import V1beta1XGBoostSpec
import os

if __name__ == '__main__':

    namespace = 'kubeflow-user-guillaume-etevenard'

    name = 'xgb'

    chicago_isvc = V1beta1InferenceService(
        api_version="serving.kserve.io/v1beta1",
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=name,
            namespace=namespace
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                xgboost=(
                    V1beta1XGBoostSpec(
                        storage_uri="s3://guillaume-etevenard/models/frompipeline/xgboost/chicago",
                        protocol_version="v2"
                    )
                ),
                service_account_name='kserve-minio-sa',
                image_pull_secrets=[{'name': 'registry-secret'}]
            )
        )

    )

    KServe = KServeClient(
        config_file = os.getenv('KUBECONFIG')
    )

    KServe.create(chicago_isvc)
