# Data Pipeline

### Kubeflow implementation of pipelines and components

On the last session we discover Data Platform ecosystem.

Now we will take a closer look to the concepts and technical implementations around [Kubeflow Pipeline](https://www.kubeflow.org/docs/components/pipelines/v1/introduction).

From the [official documentation](kubeflow.org/docs/components/pipelines/v1/concepts/component/) : 

*"A pipeline component is self-contained set of code that performs one step in the ML workflow (pipeline), such as data preprocessing, data transformation, model training, and so on. A component is analogous to a function, in that it has a name, parameters, return values, and a body."*

In this session : 

- First, we will write lab code for training xgboost model on chicago data and create a tensorboard to monitor the training. [0_local_pipeline.ipynb](0_local_pipeline.ipynb)

- Then a quick notebook to understand how to create a component, and a pipeline in kubeflow Pipeline [1_create_component_and_pipeline.ipynb](1_create_component_and_pipeline.ipynb)

- Finally, a notebook to create a pipeline with our training process on chacago data, and then upgrade this pipeline with new components [2_taxi_tips_pipeline.ipynb](2_taxi_tips_pipeline.ipynb)

