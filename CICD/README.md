# CI/CD PRACTICAL SESSION

Since the beginning of this course, we learnt:

- How to developp industrial pipelines and test it
- How to create a service from a model
- How to register features, models, data in some stores / catalogs

To achieve all of this, we did some high values (developping a feature) actions and some low value actions (execute code to launch a test).

In this lab we will group and transform those low value actions so it can be automated using a version-source-control tool.

> The corrections for the exercices 1 & 2 are presents in folders :  [pipeline_builder](./pipeline_builder/) and [deploy](./deploy/)

### Create a project in gitlab

To illustrate the power of ci/cd, we added an embedded version of gitlab in the platform :

- Go to https://gitlab.course.aiengineer.codex-platform.com/ : and **use sso login!**
- Look for the group tp5 and ask to join
- Once you joined, you can create a new project, name it `xx-pipeline-builder` where xx is the first letter of the 2 parts of your username. `john-doe --> jd-pipelinebuilder`. The project need to be created inside the **tr5 group** so it will inherit variables. Choose `internal` visibility

![create](./images/create.png)

### add some mandaroty variables in Settings/CICD

add kubeflow pipeline token : `KF_PIPELINES_SA_TOKEN_PATH`

to do this, from your project page, go to `settings  /  CI/CD`, scroll to `Variables` and expand it.

Click on `Add variable`. The name of the variable is `KF_PIPELINES_SA_TOKEN_PATH`, and its type will be "file" because the value is quite long. To get the value:

- Go back to your jupyter in kubeflow
- CLick on `+  > terminal`
- write `cat $KF_PIPELINES_SA_TOKEN_PATH` and copy paste the result to the "value" field of the variable
- switch type to "file"

![cat](./images/cat.png)

![car](./images/car.png)


### Generate a personnal Token to clone / push your repos

On gitlab, go to preferences

![pref](./images/pref.png)

On the left, go to access token, give it a name, and tick `api scope`

![tok](./images/tok.png)

Click `Generate access token` and keep the value somewhere you will find it

### Get the project locally, into jupyter lab

Now we will clone the repo you just made, and start using it. you might want to move place before cloning, to avoid stacking repos. 
```shell
cd
mkdir tp5repos
cd tp5repos
``` 

Clone the repo using url (change 'xx' with your initials),  : `git clone https://{username}:{access_token}@gitlab.course.aiengineer.codex-platform.com/tp5/ge-pipeline-builder.git`

### Config git local repo

- add username : `git config --global user.name ''`
- add mail : `git config --global user.email ''`
- checkout to a local branch : `git checkout -b feature/initci`

**Congratulations, Now we are fully configured!**

## Exercice 1 : Build components and pipelines using a gitlab pipeline

Our goal is to build components and deploy it to kubeflow, automatically when we release a stable new version of code.

For this, we need some code for the components, we will use our well known taxi trips pipeline from datapipeline2 (correction is on branch correction).

### Refactor components

For each component of our pipeline (get_data, preprocess, train, predict, export), we will create a python file with imports, component code, and builder. Example here with the get_data : 

```python
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

```
we will organise those files within the project like this

```
.
└── xx-pipeline-builder/
    ├── component_source/
    │   ├── minio_get_data.py
    │   ├── minio_export_model.py
    │   ├── preprocess.py
    │   ├── xgboost_predict.py
    │   └── xgboost_train.py
    └── README.md
```
### Refactor pipeline

the pipeline will stay close  to the datapipeline2 version, but instead to create a run from a pipeline func, we will build it and create a reccuring run everyday at 16h30

```python 

######################################

#pipeline code as in upgraded pipeline from Data_Pipeline_2/3_upgraded_taxi_tips.ipynb

#####################################

### New way to build!!
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
    job_name = "XGB_chicago_upgraded_scheduled",
    cron_expression = '30 16 * * *',
    pipeline_package_path  = 'xgboost_pipeline_upgraded.yaml', 
)
```

add it to the project : 

```
.
└── xx-pipeline-builder/
    ├── component_source/
    │   ├── minio_get_data.py
    │   ├── minio_export_model.py
    │   ├── preprocess.py
    │   ├── xgboost_predict.py
    │   └── xgboost_train.py
    ├── pipeline_source/
    │   └── pipeline.py
    ├── .gitlab-ci.yml
    └── README.md
```

### Create .gitlab-ci.yml file

Gitlab has a standard and unify way to execute code related to a push. it will look for the presence of a file named `.gitlab-ci.yml` and execute it sequentialy.

In our case this `.gitlab-ci.yml` will build our components for the pipeline, we only need to provide it the paths of the files to build, the python modules needed to do the build and some contexte.

For this exercice the contexte will be **"we want to build only when the push is made to the main branch, or when a tag is created"**

```yaml

image: python:3.8

stages:
  - components
  - launch_pipeline

before_script:
  - python --version  # For debugging
  - pip install virtualenv
  - virtualenv venv
  - source venv/bin/activate
  - pip install kfp==1.6.3 numpy==1.22.0 pandas==1.0.5 minio==6.0.2 scikit-learn==0.24.2 urllib3==1.26.11


components:
  stage: ...
  only:
    refs:
      - tags
      - main
  script:
      - mkdir components
      - python component_source/minio_get_data.py
      - python ...
      - python ...
      - python ...
      - python ...

  artifacts:
    paths:
      - components/*.yaml
    
launch_pipeline:
  stage: ...
  only:
    refs:
      - tags
      - main
  script:
    python pipeline_source/pipeline.py
```
> :warning: **jupyterlab can't see any hidden file by default**. So it will be impossible to see and open from the file browser tab a file names `.something`. The **workaround** here is to rename the file before pushing it, and to nemane it back if you need to update it:

```shell
#1 create the file with a standard name so JupyterLab will see it
touch gitlab-ci.yml
#2 edit it with the code editor
#3 rename it so gitlab will reconize it (but it will became invisible for jupyter file browser)
mv gitlab-ci.yml .gitlab-ci.yml
```

### Now let see if we did everything correctly

in a terminal, track, add and commit every file we added to the project, and push it to the remote branch

```bash
git add .....
git commit -m "....."
git push origin feature/initci
```

If you check your **repo page in gitlab**, a new branch should have appeard. But if you look at `CI/CD  / pipeline `, nothing happened.

This is because we choose to trigger the CI only with :

```yaml
only:
    refs:
      - tags
      - main
```

So now, we will merge our branch `feature/initci` to `main`. For that, click on `menu / merge request   : New Merge Request`. Choose your new branch as source, and main as target.

FIll all the fields so the MR will be useful in case of collectiv work, review your work in the "changes" tab and create the merge request

![mrg](./images/mrg.png)

![new](./images/newmr.png)

![changes](./images/changes.png)

Now you can see that your Merge request is waiting for approval. go to the page, tick "squash commits" and "delete source branch" and merge the request on main.

Now your code is the official version of the repo, and published to main branch.

Then, in the project page, go to `CI/CD / Pipeline` and check the job logs

![pipeline](./images/pipeline.png)

and look and the `reccuring run` tab from Kubeflow

![recrun](./images/recrun.png)

## Excerice 2 : model deployment automated

### Create a new project for inference

this will be `xx-deploy`. in tp5 group.

Python file will be made with datapipeline2 4_serving_chicago_model.ipynb, especially imports and isvc creation!

Adjustments when you define `KserveClient` : you need to provide a path to your `KUBECONFIG` variable, so you can deploy something to your personnal namespace

```python
    KServe = KServeClient(
        config_file = os.getenv('KUBECONFIG')
    )
```

Organisation
```
.
└── xx-deploy/
    ├──  createinference.py
    ├── .gitlab-ci.yml
    └── README.md
```

Complete `.gitlab-ci.yml`

```yaml
image: python:3.8

before_script:
  - python --version  # For debugging
  - pip install virtualenv
  - virtualenv venv
  - source venv/bin/activate
  - pip install ...

components:
  only:
    refs:
      - tags
      - main
  script:
      - python ...
```


Push this to a feature branch, as in exercice 1 do a merge request to main, before merging, go to `Settings / CI/CD / Runners` and deactivate `shared runner`, this one does not have the configuratiopn to run the kind of job we need. Then merge your branch to main and check the pipelines logs.

in the end you should see a new model deployed into "models"

![model](./images/model.png)


## Go further : Build compoment into containers

**we saw in the first exercice that our pipeline only fetch yaml components files and so Kubeflow pipeline et responsble to build this components into containers. We want to delegate this build action to gitlab, to reduce pipeline execution time**.


### Create a new repo called `xx-container-builder`

Organisation : 
```
.
└── xx-container-builder/
    ├── minio_get_data/
    │   ├── component.py
    │   ├── Dockerfile
    │   └── entrypoint.sh
    ├── xgboost_train/
    │   ├── component.py
    │   ├── Dockerfile
    │   └── entrypoint.sh
    ├── ...
    ├── Dockerfile
    ├── .gitlab-ci.yml
    └── README.md
```
- The python file contains only the function and a main with an argparser with parameter function exposed.
- The Dockerfile will be in charge of managing python dependancies, copy python src and declare an entrypoint
- The entrypoint is the call to the python script catching params. 

The `.gitlab-ci.yml` file will loop on folder and build the content.

You will need to start from an image containing docker and ask for the registry 


