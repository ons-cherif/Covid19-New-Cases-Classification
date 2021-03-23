
# Covid19 New Cases Classification

Since December 2019 we have been facing a global pandemic that changed the world for ever.
The origin of Corona Virus was Wuhan-China and from there it passed from epidemic to pandemic. <br>
A lot of researchs has revealed that the infection's globalization was caused by a mutation in the spike protein of the SARS-CoV-2, that has dramatically increased its transmissibility between humans and animals.<br>
 
 Thus several lockdowns and preventions that was globally made, we faced a death rate_considered the highest ever known during the humanity existance_.<br>
 During this project we will be using a Covid19 dataset that gathers all possible information about its propagation and the globally occured damage. <br>
 This project is about training a Machine Learning model to predict new cases and their origin using Microsoft Azure ML Studio to prepare, train and deploy the best model as a webservice.
 

## Project Details
* [Project Architecture](#project-architecture)
* [Project Set Up and Installation](#project-set-up-and-installation)
* [Dataset](#dataset)
  * [Overview](#overview)
  * [Task](#task)
  * [Access](#access)
* [Automated ML](#automated-ml)
  * [Results](#results)
* [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Results](#results)
* [Model Deployment](#model-deployment)
* [Screen Recording](#screen-recording)
* [Standout Suggestions](standout-suggestions)
* [Improvements and Future Work](#improvements-and-future-work)
* [References](#references)

## Project Architecture

*Figure 1 : The following diagram shows the overall architecture and workflow of the project.*

![](screenshots/Project_Architecture.png)


## Project Set-Up and Installation

Being familiar with Microsoft Azure and how it works, it was pretty simple to run this project.<br>

Since it's a Udacity project, there is an existing Microsoft subscription `Udacity CloudLabs Sub - 15` and a resource group `aml-quickstarts-141033`, but for my case I used my personal Microsoft Azure. 

We can resume this envioronement set up to three major sections:<br>

### Create a Workspace:

A workspace is a top-level resource needed to use all services within Microsoft Machine Learning. As defined within Microsoft's documentation: <br>
> A Workspace is a fundamental resource for machine learning in Azure Machine Learning. You use a workspace to experiment, train, and deploy machine learning models. <br>
> Each workspace is tied to an Azure subscription and resource group and has an associated SKU.
> -- <cite>[Microsoft Azure Worksapce][1]</cite>

I used the same documentation to create a workspace, but if we were using Udacity's Account, there is a pre-created workspace named `quick-starts-ws-141033`.

### Set up a Compute Instance:

Once we created a workspace to hold all our experiments and trained models, we need to create a compute instance to run our scripts using Notebboks under Studio ML.
A compute Instance helps data science community to use pre-built Azure Machine Learning functionalities, manage security and performance. 

Microsoft defines a compute instance as: <br>
> A compute instance is a fully managed cloud-based workstation optimized for your machine learning development environment.<br>
> Azure Machine Learning compute instance enables you to author, train, and deploy models in a fully integrated notebook experience in your workspace.<br>
> -- <cite>[Microsoft Azure Compute Instance][2]</cite>


We just need to install required libraries already been gathered within the requirements.txt file under the starter_file directory. I used the Microsoft Azure ML Studio embedded Terminal to run `pip install -r requirements.txt` and install the desired packages. <br>

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
### Overview

Automated machine learning is the process of automating the time consuming, iterative tasks of machine learning model development. 

AutoML is used to automate the repetitive tasks by creating a number of pipelines in parallel that try different algorithms and parameters. This iterates through ML  algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is considered to fit the   data. This process terminates when the exit criteria defined in the experiment is satisfied.

### AutoML Configuration

Instantiate an AutoMLConfig object for AutoML Configuration.

The parameters used here are:

* `n_cross_validation = 3` : Since our dataset is small. We apply cross validation with 3 folds instead of train/validation data split.
* `primary_metric = 'accuracy'` : The primary metric parameter determines the metric to be used during model training for optimization. Accuracy primary metric is chosen for binary classification dataset.
* `experiment_timeout_minutes = 30` : This defines how long, in minutes, our experiment should continue to run. Here this timeout is set to 30 minutes.
* `max_concurrent_iterations = 4` : To help manage child runs and when they can be performed, we match the number of maximum concurrent iterations of our experiment to the number of nodes in the cluster. So, we get a dedicated cluster per experiment.
* `task = 'classification'` : This specifies the experiment type as classification.
* `compute_target = cpu_cluster` : Azure Machine Learning Managed Compute is a managed service that enables the ability to train machine learning models on clusters of Azure virtual machines. Here compute target is set to cpu_cluster which is already defined with 'STANDARD_D2_V2' and maximum nodes equal to 4.
* `training_data = train_data` : This specifies the training data to be used in this experiment which is set to train_data which is a part of the dataset uploaded to the datastore.
* `label_column_name = 'DEATH_EVENT'` : The target column here is set to DEATH_EVENT which has values 1 if the patient deceased or 0 if the patient survived.
* `featurization= 'auto'` : This indicates that as part of preprocessing, data guardrails and featurization steps are performed automatically.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.


## References:

[1]: https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace.workspace?view=azure-ml-py
[2]: https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance
