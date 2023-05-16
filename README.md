# Machine Learning Project - Sagemaker

#### Table of contents
[[_TOC_]]


## 1. Introduction

This repository has been created to help and guide the different Data teams to use AWS Sagemaker and be able to develop and deploy Machine Learning models in production in an agile and efficient way. In this guide you will find almost everything you need to know to use this service, from an overview of the framework, through the Jupyter notebook adapted to Sagemaker (with some tricks), to the Pipelines to deploy models in production.

Clarify that this template is a guide made with the purpose of helping teams, if any team wants to work in a different way, with a different project structure and with their own guidelines, they are free to do it.

## 2. Sagemaker overview

### How does it work?

![Sagemaker framework](src/sagemaker_architecture.jpg)

Let’s first understand how Jupyter notebooks are set up and accessed. Jupyter notebooks are by default hosted on a single machine and can be accessed via any web browser. The following diagram illustrates how it works if set up on an Amazon Elastic Compute Cloud (Amazon EC2 instance).

You can access the Jupyter notebooks by opening the browser and entering the URL of the Jupyter server, which makes an HTTPS/WSS call to the machine where the Jupyter server is hosted. The machine runs a notebook server that receives the request and uses zeromq to communicate with the kernel process.

As commented before, each user has his own EBS volume to store his data, the only thing that is shared is the instance where the notebook server is located between different verticals.

### Setup

![Sagemaker framework](src/setup/setup_screen.png)

The interface prior to using Sagemaker is shown below. Here are all the users, where the name nomenclature refers to the vertical, followed by a hyphen with the user's name. Each user has its own filesystem and shares with the other users the EC2 instance explained above to launch Sagemaker Studio. If we want to launch Sagemaker Studio with our user, we simply click on Launch and then Studio. It is important not to use the Canvas or TensorBoard service as it is quite expensive.

### Framework

In the picture above you can see the main Sagemaker interface. We have marked with numbers what we think is most important on a day-to-day basis and we will describe each of the points below:

![Sagemaker framework](src/setup/interface/general_interface.png)

1. Filesystem in Elastic Block Storage (EBS) where all data accessible by Sagemaker instances or notebook will be stored. It is recommended to store code, small files and configuration files here. For large files it is recommended to use s3.

2. In this tab will appear the instances that are active, the apps running in these instances, the kernels and the open terminals. It is important to look at it from time to time to avoid unnecessary expenses and to use the shutdown button in case you want to delete it.

3. To clone repositories or start a new one. In general, it is recommended to work with gitlab, although other types of project repositories can be used. To clone a repository it is recommended to use the terminal and to do it by means of commands

4. This tab is used for help, extensions or themes. It is rarely used.

5. Summary of Sagemaker internal tabs in use or open.

6. To initialize a jump start, i.e. a guided machine learning project. Its use has not been investigated.

7. Summary of the different projects, clusters and other active services in Sagemaker. Above all, it will be used to monitor and check the status of the pipelines.

8. Create a new launcher

9. Create a new folder

10. Import an external file from your local server

11. Update filesystem information and view

12. Using gitlab (command line recommended)

13. To create a new instance of jupyter notebook in the directory you are in

14. To create and use a console with python3

15. Used as a terminal image of the main Sagemaker instance.

16. Used as a system terminal of the main Sagemaker instance

17. To create a .txt file

18. We have never used it, but it is used to create less ad-hoc and more guided machine learning projects, with less flexibility.

### Terminal

There are a number of commands (apart from the typical bash commands) that will make your life easier when moving around in sagemaker:

- delete-pipeline: if you want to delete a pipeline already created, with this command you can delete it, but only if it belongs to your vertical.

- list-apps: in case you want to see the applications currently in use, this is a good option.

- update-pipeline: in case you want to update a pipeline already created.

- update-pipeline-execution: if you want to update an execution of a pipeline already created.

### Gitlab

GitLab is an important tool used alongside SageMaker for version control and collaboration. It is a web-based DevOps lifecycle tool that provides a Git repository manager providing a wiki, issue-tracking, and CI/CD pipeline features.

#### Usage with SageMaker
In a typical workflow, your codebase resides in GitLab repositories. You can clone these repositories to your SageMaker environment, work on your models, experiment, and then push your changes back to GitLab.

To clone a repository into your SageMaker environment, you'd typically use the terminal within SageMaker. Here are the steps:

Open a terminal in SageMaker.
Navigate to the directory where you want to clone the repository.
Run the command: git clone [repository_url]
After making changes to your code or notebooks, you would want to commit these changes and push them back to GitLab. Here's how you do it:

Open a terminal in SageMaker.
Navigate to the directory containing the cloned repository.
Run the command: git add . to stage all changes.
Run the command: git commit -m "Your commit message" to commit the changes.
Finally, run the command: git push to push your changes to the GitLab repository.

#### Note
It is recommended to work with GitLab primarily due to its seamless integration with SageMaker and other AWS services. However, it is important to note that other version control systems can be used as well, depending on your specific project needs and requirements.

### Pipeline

In the context of Amazon SageMaker, a pipeline represents a series of machine learning (ML) steps, such as data preparation, model training, model evaluation, and deployment. Each of these steps is encapsulated as a job. Pipelines bring an order of execution to these jobs, facilitating the process of building, training, and deploying a model.

#### Creating a Pipeline
Creating a pipeline in SageMaker involves defining the steps that your machine learning workflow will go through. This could include:

- Data processing: Cleaning and transforming raw data into a format suitable for model training.

- Model training: Setting up and running a training job with your chosen algorithm and training data.

- Model evaluation: Evaluating the model's performance using a validation dataset.

- Model deployment: Deploying the trained model to an endpoint for making predictions.

Each step in the pipeline corresponds to a job that is defined using SageMaker's Python SDK. Once the pipeline is defined, it can be executed with a single command.

#### Updating a Pipeline
Once a pipeline has been created, you might need to update it to reflect changes in the workflow. For instance, you might want to use a different preprocessing method, or you might want to experiment with different training algorithms.

To update a pipeline, you can use the update-pipeline command. This allows you to change the pipeline definition and apply the changes without having to recreate the entire pipeline.

#### Deleting a Pipeline
If you no longer need a pipeline, you can delete it to clean up your SageMaker environment. This is done using the delete-pipeline command. Note that this will only delete the pipeline; any models trained by the pipeline will not be deleted.

#### Executing a Pipeline
To execute a pipeline, you can use the start-pipeline-execution command. This will run each step in the pipeline in the order they are defined.

If you want to update an execution of a pipeline that is already created, you can use the update-pipeline-execution command.

#### Monitoring a Pipeline
You can monitor the status of your pipelines through the SageMaker interface. This provides visibility into the progress of your pipeline execution and helps you identify any issues that might occur during the execution.

#### Debugging a Pipeline
In the event that a pipeline execution fails or produces unexpected results, SageMaker provides tools for debugging. Logs for each pipeline step are accessible through the AWS CloudWatch service, where you can examine detailed logs for each job in the pipeline. These logs can provide valuable insight into issues such as errors in your code or problems with data.

#### Pipeline Best Practices
- Modularity: Define each step in your pipeline as a separate, self-contained unit of work. This improves readability and makes it easier to update or replace individual steps.

- Version Control: Use a version control system such as GitLab to track changes to your pipeline definitions. This enables you to experiment with different configurations and roll back to a previous version if needed.

- Resource Management: Regularly review your active pipelines and delete any that are no longer needed. This helps manage resource usage and control costs.

- Security: Ensure that your pipelines are secure. Restrict access to your pipelines to only those who need it, use IAM roles to manage permissions, and encrypt sensitive data.

#### Pipeline and SageMaker Studio
SageMaker Studio provides a visual interface for creating and managing pipelines. You can drag and drop steps to create a pipeline, and then monitor the execution of each step in real-time. Additionally, you can view the code that defines each step, making it easier to understand what's happening at each stage of your machine learning workflow.

In summary, SageMaker pipelines are a powerful tool for orchestrating your machine learning workflows. They help you automate and standardize processes, improve collaboration, and increase the efficiency and reproducibility of your projects. By understanding how to create, update, and manage pipelines, you can take full advantage of the capabilities that SageMaker provides.


### Filesystem

The file system in Amazon SageMaker represents the structure and location of data files used in your machine learning workflows. This is typically managed using Amazon Elastic Block Store (EBS), a high-performance block storage service designed for use with Amazon EC2 for both throughput and transaction-intensive workloads at any scale.

#### Using EBS with SageMaker
Amazon EBS provides raw block-level storage that can be attached to Amazon EC2 instances, and is used in Amazon SageMaker to provide storage for your Jupyter notebooks, data files, and other important files. This allows the data to be easily accessible by your SageMaker instances or notebooks.

Storing code and small files: It is recommended to store your code files, small datasets, and configuration files in EBS. These files can be quickly accessed and used in your SageMaker notebooks.

Updating filesystem information and viewing: You can view and update your file system directly from your SageMaker notebook instance. This provides an easy way to manage your files and directories.

#### Using S3 for Large Files
For larger files and datasets, it is recommended to use Amazon S3 (Simple Storage Service), an object storage service that offers industry-leading scalability, data availability, security, and performance.

This means that you can store and retrieve any amount of data at any time, from anywhere on the web, and only pay for what you use. In your SageMaker environment, you can easily read data from S3 into your notebooks for analysis or model training, and write results or models back to S3.

#### Creating New Folders and Files
Within your SageMaker environment, you can create new directories to organize your files as you would in any other file system. Additionally, you can create new files such as .txt files or new instances of Jupyter notebooks directly within the directory you are in.

#### Importing External Files
To import external files from your local server, you can use the 'Upload' button in the SageMaker interface. This will allow you to select files from your local machine and upload them to your SageMaker environment.

In conclusion, the file system in SageMaker, along with EBS and S3, provides a robust and flexible structure for managing your data and code. Understanding how to effectively use these resources is essential for maintaining an organized and efficient workflow in your machine learning projects.

## 3. Sagemaker Notebooks [Jupyter]

Amazon SageMaker Notebook is a cloud-integrated Jupyter notebook service designed to simplify development and code execution on Amazon SageMaker. It enables developers and data scientists to create and run Jupyter notebooks in a secure, managed environment, with access to AWS resources and processing power.

SageMaker notebooks are based on Amazon Elastic Container Service (ECS) and run on dedicated instances. SageMaker manages the underlying infrastructure, which means users can focus on code and data, rather than worrying about infrastructure configuration.

SageMaker notebooks also offer seamless integration with other SageMaker features, such as data warehousing and training and deployment services. Users can access AWS resources, such as S3, EC2 and ECR, from within the notebook, allowing them to use AWS tools and services to build and train machine learning models and create comprehensive data analytics workflows.

In short, the AWS SageMaker notebook environment is a secure, managed cloud development platform for building, running and distributing machine learning and data analytics applications.

//TODO: FOTOS, COMENTARIOS DEL LAYOUT, ETC

### Framework

The framework in Amazon SageMaker provides an integrated and scalable environment for handling all stages of a machine learning workflow - including data exploration, data cleaning, model training, model evaluation, and deployment.

#### Understanding the SageMaker Interface
The SageMaker interface is intuitive and user-friendly, designed to help you manage your machine learning projects efficiently. Here are some of the main components of the SageMaker interface:

- Instances: The instances tab shows all active instances, including the apps running on these instances, the kernels, and the open terminals. Regularly checking this tab will help avoid unnecessary expenses. You can shut down instances when they are no longer needed.

- Repositories: SageMaker provides the ability to clone repositories or start a new one. It is generally recommended to use GitLab for version control, but other repository services can also be used.

- Help and Extensions: This tab is used for accessing help resources, managing extensions, and changing themes.

- Internal Tabs: A summary of SageMaker internal tabs that are in use or open can be seen, allowing for easy navigation between different components of SageMaker.

- Jump Start: This option allows you to initialize a guided machine learning project.

- Projects and Services: This provides a summary of the different projects, clusters, and other active services in SageMaker. It is mainly used to monitor and check the status of the pipelines.

#### Creating New Resources
From the SageMaker interface, you can also perform actions such as creating a new launcher, a new folder, or importing an external file from your local server. You can also create new instances of Jupyter notebooks in the directory you are in or create and use a console with Python3.

#### Working with the Terminal
The terminal in SageMaker serves as a command-line interface where you can run a variety of commands to manage your SageMaker resources. Some of the commands you may frequently use include delete-pipeline, list-apps, update-pipeline, and update-pipeline-execution.

In conclusion, the SageMaker framework provides a comprehensive, user-friendly, and powerful platform for machine learning development. By understanding the various components and how they interact, you can leverage SageMaker to streamline your machine learning workflows and enhance productivity.

### Instances

AWS SageMaker offers several main instances that can be used depending on the use case. It is important to emphasize that due to infra limitations, not all of them can be used and in case of not being able to use it, an error will be displayed. The type instances are the following:

- ml.t2.medium: This is a low-cost instance that can be used for development and testing purposes.

- ml.m4.xlarge: This is a more powerful instance that can be used for training and intensive processing tasks. We recommend this instance for running processes with Pandas and for steps in production for both in-memory processing and training. If we want more capacity or memory, we simply increase the number (i.e. ml.m8.xlarge).

- ml.p2.xlarge: This is an instance designed for GPU processing tasks, such as training deep learning models.

- ml.g4dn.xlarge: This is a GPU instance designed for graphics-intensive tasks, such as data visualization and image generation.

- ml.c5.large: An instance designed for processing-intensive tasks that require high network bandwidth, such as large-scale data transfer.

Use case recommendation for each instance:

- ml.t2.medium: Ideal for testing and code development. We recommend this instance for launching processes in pyspark, as the instance is only used to connect to the EMR cluster and does not require high capacity and memory.

- ml.m4.xlarge: Ideal for model training and intensive processing.

- ml.p2.xlarge: Ideal for deep learning model training and other GPU processing tasks.

- ml.g4dn.xlarge: Ideal for data visualization and image generation.

- ml.c5.large: Ideal for large-scale data transfer and other processing-intensive tasks requiring high network bandwidth.

These are only general recommendations. The final use of each instance will depend on the specific use case and the performance and cost requirements. The choice of the appropriate instance depends on many factors, such as the size of the model, the size of the training data, the desired speed and efficiency, among others. It is important to carefully evaluate the needs and requirements before selecting an instance for use with Amazon SageMaker.

To select a machine, you must perform the following steps

![How to select image for notebook ](src/setup/instances/HowToSelectMachine.png)

Once selected, a menu will be displayed in which you can filter the machines that have quick access. It should be noted that the access times to any machine is affordable, being only a few minutes in most cases.

![Menu for machine selection](src/setup/instances/MachinesAllow.png)

To check the prices associated with each of the machines that can be selected, you can go to the following address: https://aws.amazon.com/es/sagemaker/pricing/

### Memory/CPU based

In Amazon SageMaker, as in other cloud-based environments, resources such as memory and CPU play a crucial role in determining the performance of your machine learning workflows. Understanding how these resources are used and allocated can help you optimize your workflows and manage costs.

#### Memory in SageMaker
Memory is a critical resource for machine learning tasks, as these tasks often involve handling large amounts of data and require significant memory capacity. In SageMaker, the amount of memory available to you depends on the type of EC2 instance you choose for your notebook instances or training jobs. Amazon provides a wide variety of instance types with varying amounts of memory to cater to different requirements.

Large datasets or memory-intensive operations might require instances with a high memory capacity. SageMaker also allows for distributed training on multiple instances, effectively multiplying the available memory.

#### CPU in SageMaker
The Central Processing Unit (CPU) is responsible for executing the instructions of a computer program. In machine learning, CPUs are essential for data processing and model training, especially for tasks that can't be easily parallelized.

The number of CPU cores available in SageMaker depends on the EC2 instance type chosen. Some instance types come with powerful CPUs with multiple cores, which can speed up the training process significantly.

#### Choosing the Right Instance Type
Selecting the right instance type for your SageMaker tasks is a balance between performance and cost. High-memory and high-CPU instances provide better performance but come at a higher cost. You should consider the requirements of your specific tasks and datasets when selecting an instance type.

#### Monitoring Usage
SageMaker provides tools for monitoring the usage of memory and CPU resources. Regular monitoring can help you identify bottlenecks, understand the performance of your tasks, and make necessary adjustments.

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

##### Tips

- Understand Your Resource Needs: Before starting a project, understand the memory, CPU, and storage requirements of your tasks. This will help you choose the right instance type and avoid unnecessary costs.

- Use the Right Tools for the Job: Use EBS for storing code and small files, and S3 for large datasets. Use SageMaker's built-in algorithms and models when possible to save time and effort.

- Leverage Built-in Capabilities: SageMaker comes with a range of built-in capabilities, such as automatic model tuning, distributed training, and managed endpoints. Make sure to leverage these features to streamline your machine learning workflows.

- Monitor Your Resources: Regularly monitor your resource usage to avoid unnecessary costs. Use CloudWatch to keep track of your instance performance and adjust as needed.

- Clean Up Unused Resources: Regularly review and delete unused resources, such as notebook instances, training jobs, and endpoints. This will help you avoid unnecessary costs.

- Use Version Control: Use a version control system like GitLab to manage your code. This allows you to track changes, collaborate with others, and revert back to previous versions if needed.

- Secure Your Work: Use AWS's security best practices to protect your data and models. This includes using IAM roles for access control, encrypting sensitive data, and regularly reviewing your security settings.

- Stay Updated: AWS regularly updates SageMaker with new features and improvements. Stay updated with the latest changes to make the most of your SageMaker environment.

- Use SageMaker Studio: SageMaker Studio provides a single, web-based visual interface where you can perform all ML development steps. It enhances productivity by enabling you to create, train, debug, deploy, and monitor all your machine learning models in a single place.

- Learn from Others: Make use of the extensive documentation, tutorials, and examples provided by AWS and the broader SageMaker community. These resources can provide valuable guidance and insights.

Remember, the goal is to leverage the full potential of SageMaker to create robust, efficient, and scalable machine learning solutions.

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


## 4. Production

### Project structure

#### Exploratory data analysis (EDA)

#### Develop

#### Production

### Continuous integration with GitLab

### Sagemaker Pipeline (framework and tips)

### Foto grafo + explicación de cada step y su código

# Utils for mkdocs

```
introduce code
```