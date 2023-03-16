# Machine Learning Project - Sagemaker



## Introduction

Amazon SageMaker is a fully managed service that helps developers and data engineers build, train, and deploy high-quality machine learning models. It offers a variety of tools and resources for companies to easily implement artificial intelligence (AI) solutions into their applications and systems.

With SageMaker, developers can access a complete set of machine learning tools in the cloud, enabling them to perform tasks such as data preparation, model building, model training and model deployment. In addition, the service also provides pre-trained, out-of-the-box machine learning models for common tasks such as image classification and natural language processing (NLP).

SageMaker integrates with a variety of other AWS services, allowing developers to take advantage of the scalability, security and reliability of the AWS cloud platform. Models trained on SageMaker can be easily deployed to web and mobile applications through AWS services such as Amazon API Gateway and AWS Lambda.

In addition, SageMaker offers a variety of model lifecycle management tools and resources that enable developers to efficiently manage their models. Developers can track the performance of their models and adjust them as needed to ensure they remain accurate over time.

Furthermore, SageMaker provides easy-to-use tools for developers to optimize and fine-tune their models for specific use cases, such as improving accuracy or reducing inference time. The service also offers a range of instance types and sizes to accommodate various workloads and budgets.

Another advantage of using SageMaker is its ability to automate and streamline the machine learning workflow, allowing developers to focus on building better models instead of worrying about infrastructure and operational tasks. For example, SageMaker can automatically launch and manage instances for data processing and model training, as well as automate hyperparameter tuning and model selection.

SageMaker also supports a variety of popular machine learning frameworks and libraries, such as TensorFlow, PyTorch, and Scikit-learn, making it easy for developers to work with their preferred tools and languages.

Overall, Amazon SageMaker provides a comprehensive set of tools and resources for developers to build, train and deploy high-quality machine learning models with ease, while taking advantage of the scalability, security and reliability of the AWS cloud platform.

This makes it a valuable service for companies of all sizes looking to implement AI solutions into their applications and systems. With SageMaker, developers can quickly and easily create custom models tailored to their specific business needs, and deploy them with confidence knowing that they are backed by the reliability and security of AWS.

In addition to its technical capabilities, SageMaker also offers extensive documentation, tutorials, and support resources to help developers get started and troubleshoot issues. This makes it accessible to developers with varying levels of experience in machine learning and AI.

Overall, Amazon SageMaker is a powerful tool for companies looking to leverage the benefits of machine learning and AI in their applications and systems. Its fully managed service, range of tools and resources, and integration with AWS services make it a reliable and efficient solution for building, training, and deploying high-quality machine learning models.

Moreover, SageMaker provides a cost-effective approach to machine learning, allowing companies to pay only for the resources they use. With SageMaker, developers can easily scale up or down their computing resources based on the size and complexity of their machine learning workloads, helping to reduce costs and optimize performance.

SageMaker also provides built-in security features to help protect data and models, such as encryption at rest and in transit, identity and access management, and compliance with industry standards and regulations.

Finally, SageMaker is constantly evolving and expanding, with new features and capabilities being added regularly. This ensures that developers have access to the latest machine learning tools and technologies, and can continue to improve and innovate their models over time.

In summary, Amazon SageMaker is a valuable service that provides developers and data engineers with the tools and resources needed to build, train, and deploy high-quality machine learning models. Its integration with AWS services, ease of use, and cost-effectiveness make it a compelling choice for companies looking to implement AI solutions into their applications and systems.

## Setup

### Framework

### Commands

### Terminal

### Gitlab

### Pipeline

### Filesystem

## Sagemaker Notebooks [Jupyter]

Amazon SageMaker Notebook is a cloud-integrated Jupyter notebook service designed to simplify development and code execution on Amazon SageMaker. It enables developers and data scientists to create and run Jupyter notebooks in a secure, managed environment, with access to AWS resources and processing power.

SageMaker notebooks are based on Amazon Elastic Container Service (ECS) and run on dedicated instances. SageMaker manages the underlying infrastructure, which means users can focus on code and data, rather than worrying about infrastructure configuration.

SageMaker notebooks also offer seamless integration with other SageMaker features, such as data warehousing and training and deployment services. Users can access AWS resources, such as S3, EC2 and ECR, from within the notebook, allowing them to use AWS tools and services to build and train machine learning models and create comprehensive data analytics workflows.

In short, the AWS SageMaker notebook environment is a secure, managed cloud development platform for building, running and distributing machine learning and data analytics applications.

//TODO: FOTOS, COMENTARIOS DEL LAYOUT, ETC

### Framework

### Instances

AWS SageMaker offers several main instances that can be used depending on the use case:

ml.t2.medium: This is a low-cost instance that can be used for development and testing purposes.

ml.m4.xlarge: This is a more powerful instance that can be used for training and intensive processing tasks.

ml.p2.xlarge: This is an instance designed for GPU processing tasks, such as training deep learning models.

ml.g4dn.xlarge: This is a GPU instance designed for graphics-intensive tasks, such as data visualization and image generation.

ml.c5.large: An instance designed for processing-intensive tasks that require high network bandwidth, such as large-scale data transfer.

Use case recommendation for each instance:

ml.t2.medium: Ideal for testing and code development.

ml.m4.xlarge: Ideal for model training and intensive processing.

ml.p2.xlarge: Ideal for deep learning model training and other GPU processing tasks.

ml.g4dn.xlarge: Ideal for data visualization and image generation.

ml.c5.large: Ideal for large-scale data transfer and other processing-intensive tasks requiring high network bandwidth.

These are only general recommendations. The final use of each instance will depend on the specific use case and the performance and cost requirements. The choice of the appropriate instance depends on many factors, such as the size of the model, the size of the training data, the desired speed and efficiency, among others. It is important to carefully evaluate the needs and requirements before selecting an instance for use with Amazon SageMaker.

To select a machine, you must perform the following steps

![How to select image for notebook ](https://gitlab.com/iberia-data/data-science/sagemaker-template/-/blob/main/src/setup/instances/HowToSelectMachine.png)

Once selected, a menu will be displayed in which you can filter the machines that have quick access. It should be noted that the access times to any machine is affordable, being only a few minutes in most cases.

![Menu for machine selection](https://gitlab.com/iberia-data/data-science/sagemaker-template/-/blob/main/src/setup/instances/MachinesAllow.png)

To check the prices associated with each of the machines that can be selected, you can go to the following address: https://aws.amazon.com/es/sagemaker/pricing/


#### Memory/CPU based

##### Installing libraries

Installation of libraries on a SageMaker Notebook machine:

Open a terminal on the Jupyter notebook.
Use the !pip install command to install the required libraries. For example, !pip install pandas matplotlib.
Note that some libraries, such as the data analysis libraries, may require more memory and processing power than is provided by default on a SageMaker Notebook machine. In these cases, it may be necessary to increase the memory and processing power of the machine before installing the libraries.

The following problems may be encountered when attempting to install libraries on SageMaker:

Insufficient permissions: If the user does not have sufficient permissions to install libraries in the SageMaker environment, they may receive an error message indicating that they do not have permissions to install libraries.

Dependency conflicts: Libraries often depend on other libraries to function properly. If a library depends on a specific version of another library, but the currently installed version is different, a dependency conflict may occur resulting in an error when installing the library.

Internet connection problems: If the SageMaker environment does not have Internet access, it may not be able to download the required libraries during installation.

Compatibility issues: Libraries are often developed to work with specific versions of Python and other technologies. If a library is not compatible with the version of Python or technology running in the SageMaker environment, an error may occur when attempting to install the library.

Resource issues: Some libraries may require a large amount of resources to function properly. If the SageMaker environment does not have sufficient resources available, an error may occur when attempting to install the library.

These are some of the most common problems a user may encounter when trying to install libraries in SageMaker. To resolve these problems, it is important to investigate the error message, review the documentation and consult the user community.

Additionally, it is recommended to check for updates and compatibility with the latest versions of libraries and technologies before installing. It may also be helpful to use a virtual environment or container to isolate the installation of libraries from other projects.

Finally, to avoid potential issues with library installation in SageMaker, it is a best practice to document the libraries and their versions used in a project, and to save them in a requirements.txt file. This can help ensure that the same libraries and versions are used consistently across different environments and projects.

In summary, installing libraries in SageMaker can be a straightforward process using the !pip install command, but it is important to be aware of potential issues such as insufficient permissions, dependency conflicts, internet connection problems, compatibility issues, and resource issues. By understanding these issues and following best practices, developers can successfully install the required libraries and build high-quality machine learning models in SageMaker.

##### Tips

#### Distributed based [EMR]

##### Installing libraries

Installing libraries on an Apache Spark cluster and on a SageMaker Notebook machine is a different process, depending on how Spark is being used.

Installing libraries on a Spark cluster:

Create a Python dependencies file, such as a requirements.txt file, containing the libraries you want to install.
Upload the dependencies file to S3 or other Spark-accessible file storage.
Start a Spark cluster on Amazon EMR, which will include an option to specify the location of the dependency file.
Once the cluster is active, you can install the libraries on the cluster by using !pip install -r and providing the location of the dependency file in S3.

Another option available to the user is the use of the command//TODO

##### SparkMagic

SparkMagic is a set of commands that allow users to run code on an Apache Spark cluster from within a Jupyter notebook. These commands are designed to facilitate the integration of Spark with Jupyter and provide a command line interface for interacting with Spark from the notebook.

With SparkMagic, users can run code in Spark from the notebook without having to leave the Jupyter interface. In addition, SparkMagic also allows users to visualize and explore the results of Spark commands on the notebook itself.

SparkMagic also provides seamless integration with other AWS tools and services, such as S3, EC2 and EMR. This means users can access data stored in S3 and process it with Spark, use EC2 to run the Spark cluster, and use EMR to manage the underlying infrastructure.

In short, SparkMagic commands provide a simple and efficient way to interact with Apache Spark from a Jupyter notebook, and also allow users to use AWS features and resources to process and analyze data in large volumes.

SparkMagic commands include the following:

%spark configure: This command allows users to configure the connection to a Spark cluster. Users can specify the cluster URL, port, user and password, among other details.

%spark: This command allows users to create a Spark session on the specified cluster using %spark configure. After creating the session, users can write and execute code in Spark directly from the notebook.

%spark info: This command displays information about the current Spark session, including the session ID, the cluster to which it is connected, and the session status.

%spark log: This command displays logs for the current Spark session. These logs can be useful for debugging errors and better understanding what is happening in the cluster.

%spark cleanup: This command closes the current Spark session and frees the resources associated with the session.

%spark add: This command allows users to add files to the current Spark session. Users can add local or remote files to the session so that they can be accessed and used by Spark.

%spark load: This command allows users to load data into Spark from a local or remote file.

%spark display: This command allows users to display the results of Spark commands on the notebook. The results are displayed in a user-friendly format, such as a table or a graph.

These are the main SparkMagic commands, and together they provide an efficient and easy way to interact with Spark from the Jupyter notebook.

##### Tips

Amazon EMR is a distributed data processing service that enables users to process large amounts of data using frameworks such as Apache Spark and Hadoop. Amazon SageMaker, on the other hand, is a fully managed service that helps developers create, train and deploy high-quality machine learning models. Here are some tips for using Amazon EMR from Amazon SageMaker:

Use Amazon EMR Notebooks - Amazon EMR Notebooks is a service that allows users to create and run Jupyter notebooks on an EMR cluster. This means you can run code in Spark or Hadoop on the same EMR cluster you are working on, which reduces latency and improves performance. In addition, you can create an EMR cluster directly from Amazon SageMaker and then connect to it from an EMR notebook.

Use EMR Studio: EMR Studio is a web development environment that enables users to create, develop and run big data-based applications. By using EMR Studio, users can connect directly to their EMR cluster and run code in Spark or Hadoop. EMR Studio is an ideal choice for those who prefer a graphical user interface instead of a command line.

Use the Boto3 SDK: The Boto3 SDK is a Python library that allows developers to interact with AWS services. Using the Boto3 SDK, users can connect to their EMR cluster and run code in Spark or Hadoop directly from Python. This is especially useful for those who prefer to write code in Python rather than Scala or Java.

Use SageMaker's integration with EMR: Amazon SageMaker integrates seamlessly with EMR, which means you can use EMR to preprocess your data before training your models in SageMaker. To use this integration, you need to create an EMR cluster and load your data into it. You can then use Spark or Hadoop to preprocess your data and store it in Amazon S3. Then, you can use SageMaker to train your models using the preprocessed data.

In summary, Amazon EMR and Amazon SageMaker are two powerful services that complement each other to enable users to process large amounts of data and train high-quality machine learning models. By using the tips above, you can take full advantage of these tools and improve the efficiency of your big data and machine learning workflows.

TODO: Meter tips mas especificos

### Plotting

In Amazon SageMaker, users can generate graphs and visualizations to explore and understand their training data and machine learning models. This can be done using different visualization tools and libraries, such as Matplotlib, Seaborn, Plotly, among others.

However, when generating charts in Amazon SageMaker, users may encounter some issues, such as:

Configuration of visualization packages: Ensuring that the packages needed to generate charts are installed and configured correctly in the Amazon SageMaker environment can be a challenge for some users.

Performance issues: Generating charts with large amounts of data can require a lot of processing resources and can cause performance issues in the Amazon SageMaker environment.

Display issues: Sometimes, generated charts may not look as expected due to formatting or configuration issues.

Therefore, it is important for users to be aware of these potential issues when generating charts in Amazon SageMaker and take steps to address them. This includes documenting and using the appropriate visualization tools and libraries, as well as optimizing processing resources and configuring the necessary packages correctly.

Depending on the tool or library you want to use, here are some examples:

Matplotlib: is an open source data visualization library that can be used to generate charts in Amazon SageMaker. You can import and use Matplotlib in an Amazon SageMaker notebook and use its functions to generate charts.

```
!pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np

# We generate example data
data = [3, 4, 1, 1, 5, 2]

# Create bar chart
plt.bar(np.arange(len(data)), data)

# Show chart
plt.show()
```

This code first installs the Matplotlib package, then imports the library and creates a bar chart with example data. The plt.show() function displays the chart on the notebook.

Seaborn: is another open source data visualization library that can be used to generate charts in Amazon SageMaker. Seaborn provides an easy-to-use interface for creating complex charts using Matplotlib.

```
!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# We generate example data
data = np.random.rand(50, 2)

# Create scatterplot
sns.scatterplot(data[:, 0], data[:, 1])

# Show plot
plt.show()
```

This code first installs the Seaborn package, then imports the library and creates a scatterplot with sample data. The plt.show() function displays the plot on the notebook.

Plotly: is an open source interactive visualization library that can be used to generate charts in Amazon SageMaker. With Plotly, users can create interactive charts with a wide variety of features and styles.

```
!pip install plotly
import plotly.express as px
import pandas as pd

# We generate example data
data = {'Fruit': ['Apple', 'Banana', 'Cherry', 'Date'],
        'Count': [3, 4, 1, 5]}

# Create DataFrame
df = pd.DataFrame(data)

# Create bar chart
fig = px.bar(df, x='Fruit', y='Count')

# Show graph
fig.show()
```

This code first installs the Plotly package, then imports the library and creates a bar chart with example data in a Pandas DataFrame. The fig.show() function displays the chart in the notebook.

To use any of these tools or libraries, users must first install and configure the necessary packages in their Amazon SageMaker environment. Then, they can import the library into their notebook and use its functions to generate charts.

It is important to note that chart generation can require a lot of processing resources and can cause performance issues in the Amazon SageMaker environment, so it is important to optimize resources and properly configure the necessary packages.

### boto3 [s3]

Boto3 is a software library developed by Amazon Web Services (AWS) that provides an application programming interface (API) for interacting with different AWS services. Boto3 makes it easy to work with AWS services from Python.

In Sagemaker, Boto3 is used to manage and automate tasks related to the creation and use of machine learning models. For example, Boto3 can be used to create and manage Sagemaker resources, such as training and production instances, and to send and receive data to and from Sagemaker.

Boto3 is built on top of a library called Botocore, which it shares with AWS CLI. Botocore provides the low-level client, session, and credential and configuration data. Boto 3 builds on Botocore by providing its own session, resources, and collections.

Modules generally fall into two categories, those that include a high-level object-oriented interface and those that include only a low-level interface that matches the Amazon Web Services API (These interfaces are resources and clients respectively. a look at both shortly).
Some modules are completely high-level (such as Amazon S3 or EC2), some include high-level code over a low-level connection (such as Amazon DynamoDB), and some are 100% low-level (such as Amazon Elastic Transcoder).
Clients: are low-level service connections with the following main characteristics:

Supports all services.
Outputs are returned using Python dictionaries.
We have to traverse these dictionaries ourselves
Automatic pagination of responses (Paginators)
A way to block until a certain state has been reached (Waiters)

Along with these core features, Boto 3 also provides sessions and per-session credentials and configuration, as well as basic components such as authentication, parameter and response handling, an event system for customizations, and logic for retrying failed requests.

Some code examples of using Boto3 with Sagemaker are described below:

Read files present with pandas from S3:
```
import boto3

s3 = boto3.client('s3')

response = s3.list_objects(
    Bucket='my-bucket',
    Prefix='path/to/my/files/'
)

files = [content['Key'] for content in response.get('Contents', [])]

print(files)
```

Create a Sagemaker trainer:

```
import boto3

sagemaker = boto3.client('sagemaker')

training_image = '012345678910.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest'
role = 'arn:aws:iam::012345678910:role/service-role/AmazonSageMaker-ExecutionRole-20181129T121160'

response = sagemaker.create_training_job(
    TrainingJobName='my-training-job',
    AlgorithmSpecification={
        'TrainingImage': training_image,
        'TrainingInputMode': 'File'
    },
    RoleArn=role,
    InputDataConfig=[
        {
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://my-bucket/train',
                    { 'S3DataDistributionType': 'FullyReplicated'.
                }
            },
            'ContentType': 'application/x-record',
            'CompressionType': 'None'
        }
    ],
    OutputDataConfig={
        'S3OutputPath': 's3://my-bucket/output'
    },
    ResourceConfig={
        'InstanceType': 'ml.c4.xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 50
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 3600
    },
    HyperParameters={
        'image_shape': '3,224,224',
        'num_classes': '2',
        'num_training_samples': '15420',
        'mini_batch_size': '10',
        'epochs': '10',
        'learning_rate': '0.01'
    }
)

```
Create a Sagemaker endpoint:

```
import boto3

sagemaker = boto3.client('sagemaker')

response = sagemaker.create_endpoint(
    EndpointName='my-endpoint',
    EndpointConfigName='my-endpoint-config', EndpointConfigName='my-endpoint-config'.
)

print(response)
```
In this example, an endpoint with the name my-endpoint is created and associated with a previously created endpoint configuration named my-endpoint-config. The API response will include information about the created endpoint, such as the ID and the current status.

Note: Before creating an endpoint, you must first create an endpoint configuration that specifies details about how the model will be used in production.


## Production


### Project structure

#### Exploratory data analysis (EDA)

#### Develop

#### Production

Before deploying your model to Amazon SageMaker, it's essential to prepare it properly. In this section, we will cover the required steps in detail, along with code examples.

2.1. Train and Save Your Model

First, you need to train your model using your preferred machine learning framework (e.g., TensorFlow, PyTorch, or scikit-learn). After training, save your model in the required format for SageMaker. Most frameworks have built-in functions to save the model as a file.

Example using TensorFlow:

```
import tensorflow as tf

# Define and train your model
model = tf.keras.Sequential([...])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# Save your trained model
model.save('my_model.h5')

```

2.2. Convert Model to a SageMaker-Compatible Format

Once your model is trained and saved, convert it to a format that can be used with SageMaker. For TensorFlow and PyTorch models, you can save the model as a .tar.gz file, while scikit-learn models need to be saved as a .joblib file.

Example using TensorFlow:

```
import tarfile

# Compress the TensorFlow model into a .tar.gz file
with tarfile.open('my_model.tar.gz', 'w:gz') as tar:
    tar.add('my_model.h5')
```

2.3. Create an Entry Point Script

Create a script that serves as an entry point for your model. This script should include functions for loading the model, preprocessing input data, predicting with the model, and post-processing the output. Save this script as a separate file (e.g., inference.py).

Example using TensorFlow:

```
# inference.py

import json
import tensorflow as tf

_MODEL = None

def model_fn(model_dir):
    global _MODEL
    _MODEL = tf.keras.models.load_model(f'{model_dir}/my_model.h5')
    return _MODEL

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError('Content type must be application/json')

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction.tolist()), response_content_type
    else:
        raise ValueError('Content type must be application/json')
```

Creating a SageMaker Pipeline
SageMaker Pipelines allows you to create, automate, and manage end-to-end machine learning workflows, from data preprocessing and model training to deployment.

3.1. Define Pipeline Steps

Start by defining the steps for your pipeline, such as data preprocessing, model training, and model deployment. Each step is defined using a SageMaker Python SDK class.

Example using SageMaker Python SDK:

```
import boto3
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.pipeline import Pipeline

# Define a data preprocessing step
sklearn_processor = SKLearnProcessor(framework_version='0.23-1',
                                     role=<your_sagemaker_role>,
                                     instance_type='ml.m5.xlarge',
                                     instance_count=1)

preprocessing_step = ProcessingStep(
    name='DataPreprocessing',
    processor=sklearn_processor,
    inputs=[ProcessingInput(source=<input_data_s3_path>,
                            destination='/opt/ml/processing/input')],
    outputs=[ProcessingOutput(output_name='train',
                              source='/opt/ml/processing/output/train'),
             ProcessingOutput(output_name='validation',
                              source='/opt/ml/processing/output/validation')],
    code='preprocessing.py'
)

# Define a model training step
estimator = <your_sagemaker_estimator>

training_step = TrainingStep(
    name='ModelTraining',
    estimator=estimator,
    inputs={'train': sagemaker.inputs.TrainingInput(s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3)
    content_type='text/csv'),
    validation': sagemaker.inputs.TrainingInput(s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri, content_type='text/csv')
}
)

Define a model creation step
model_step = CreateModelStep(
    name='CreateModel',
    model_name=<model_name>,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    primary_container_image=<container_image>,
    role=<your_sagemaker_role>
)

Define a model deployment step
deploy_step = steps.EndpointStep(
    name='DeployModel',
    endpoint_name=<endpoint_name>,
    model_name=model_step.properties.ModelName,
    config_name=<endpoint_config_name>,
    update=False
)
```


Once you've defined the steps, assemble them into a pipeline and execute it using the SageMaker Python SDK.

Example:
```
# Assemble the pipeline
pipeline = Pipeline(
    name=<pipeline_name>,
    steps=[preprocessing_step, training_step, model_step, deploy_step]
)

# Execute the pipeline
pipeline.upsert(role_arn=<your_sagemaker_role>)
execution = pipeline.start()
execution.wait()
```

These steps will create a SageMaker Pipeline that automates your machine learning workflow, from data preprocessing, model training, to deployment. The pipeline is defined and executed using the SageMaker Python SDK, making it easy to manage and monitor your end-to-end process.

### Continuous integration with GitLab

### Sagemaker Pipeline (framework and tips)

### Foto grafo + explicación de cada step y su código

# Utils for mkdocs

```
introduce code
```