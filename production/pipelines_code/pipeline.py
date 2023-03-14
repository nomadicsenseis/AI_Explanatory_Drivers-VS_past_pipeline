"""
This is a template for a pipeline definition.

The pipeline is defined using the sagemaker Python SDK, and it consists of several steps that execute
a sequence of processing and training jobs to produce a trained model.

The pipeline can takes several parameters, including the region where the pipeline will be executed,
the name of the pipeline, a base job prefix, an IAM role, a default S3 bucket, and more user defined parameters.

The pipeline can be formed with multiple steps.
"""
import logging
from os import chdir, pardir, sep
from os.path import dirname
from os.path import join as path_join
from os.path import realpath
from typing import Optional

from production.pipelines_code import utils
from sagemaker.session import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# This is necessary to ensure that all relative paths in the project are resolved correctly and
# to avoid errors related to incorrect file references.
BASE_DIR = dirname(realpath(f"{__file__}{sep}{pardir}"))
chdir(BASE_DIR)


def set_pipeline_definition(
    region: str,
    pipeline_name: str,
    base_job_prefix: str,
    role: Optional[str] = None,
    default_bucket: Optional[str] = None,
) -> Pipeline:
    """
    Sets up an AWS SageMaker pipeline definition that can be used for both training and prediction.

    Args:
        region: The name of the AWS region to use.
        pipeline_name: The name of the pipeline to create.
        base_job_prefix: The base name of the SageMaker jobs that will be created.
        role: The AWS IAM role to use for the pipeline execution. If not provided,
            the execution role will be obtained from the AWS SageMaker session. Defaults to None.
        default_bucket: The name of the default S3 bucket to use. If not provided,
            the bucket will be obtained from the AWS SageMaker session. Defaults to None.

    Returns:
        Pipeline: The SageMaker pipeline definition.

    Raises:
        ValueError: If the provided `is_for_predict` value is not valid (i.e., not equal to "0" or "1").

    """

    # INITIALIZE THE VARIABLES
    # Sets the session and the execution role.
    sagemaker_session = utils.get_session(region=region, default_bucket=default_bucket)
    if role is None:
        role = get_execution_role(sagemaker_session)

    # These parameters are the ones that the pipeline needs to execute. They are the parameters that will be used
    # the different execution steps that are defined.
    # TIP: Although there are different types of parameters such as ParameterString or ParameterInteger,
    # for those parameters that are going to be arguments of execution steps, it is necessary that
    # they be of type ParameterString and of type ParameterString and then in the execution
    # code itself change the type if necessary.
    processing_instance_count = ParameterInteger(name="processing_instance_count", default_value=3)
    is_for_predict = ParameterString(name="is_for_predict", default_value="0")
    execution_date = ParameterString(name="str_execution_date")
    s3_bucket = ParameterString(name="s3_bucket", default_value="iberia-data-lake")
    s3_key_data_source = ParameterString(name="s3_key_data_source")
    s3_key = ParameterString(name="s3_path_write")

    # We are going to create a pipeline that can be used for both training and prediction.
    # For this we need conditions such as ConditionEquals and ConditionSteps.
    train_condition = ConditionEquals(left=is_for_predict, right="0")

    # Within this class are the different types of instances that can be raised.
    processors = utils.Processors(
        base_job_prefix=base_job_prefix,
        role=role,
        instance_count=processing_instance_count,
        instance_type="ml.m5.4xlarge",
        sagemaker_session=sagemaker_session,
    )

    # CREATE STEPS
    # One of the steps that can be used are to generate ETLs. These use PySpark, and in order to be able to
    # run them properly you first have to define the PySpark configuration. The properties
    # that can be changed can be seen here https://spark.apache.org/docs/latest/configuration.html.
    # These properties are defined in a config.yml file on the same path as this pipeline.py file.
    # Inside the config.yml file there is a key named PYSPARK, and it is under this where you can
    # define the properties that we want the machine that is going to execute PySpark code to have.
    # It is necessary to take into account that the names of the properties come with ".", these have to be substituted
    # by "_" so that there are no errors of malformation of the file.
    configuration = utils.read_config_data()
    pyspark_properties = {
        key.replace("_", "."): value
        for key, value in configuration.get("PYSPARK").items()
    }
    pyspark_config = [
        {"Classification": "spark-defaults", "Properties": pyspark_properties}
    ]

    # ETL step
    pyspark_processor = processors.pyspark()
    etl_step_args = pyspark_processor.get_run_args(
        submit_app=path_join(BASE_DIR, "code", "etl.py"),
        submit_py_files=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "etl_utils.py"),
        ],  # This argument is for passing Python files.
        submit_files=[path_join(BASE_DIR, "packages", "config.yml")],  # This argument is for passing text files.
        arguments=[
            "--s3_bucket", s3_bucket,
            "--s3_key_data_source", s3_key_data_source,
            "--s3_key", s3_key,
            "--execution_date", execution_date,
            "--is_for_predict", is_for_predict,
        ],  # This argument is to pass the arguments to the executable.
        configuration=pyspark_config,
    )
    etl_step = ProcessingStep(
        name="etl_step",
        processor=pyspark_processor,
        inputs=etl_step_args.inputs,
        outputs=etl_step_args.outputs,
        job_arguments=etl_step_args.arguments,
        code=etl_step_args.code,
    )

    # Preprocess step
    framework_processor = processors.framework()
    preprocess_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "preprocess.py"),
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "preprocess.txt")
        ],  # This argument is for passing needed files.
        arguments=[
            "--s3_bucket", s3_bucket,
            "--s3_key", s3_key,
            "--execution_date", execution_date,
            "--is_for_predict", is_for_predict,
        ],  # This argument is to pass the arguments to the executable.
    )
    preprocess_step = ProcessingStep(
        name="preprocess_step",
        depends_on=["etl_step"],  # This argument is to create links between steps. Create an execution graph.
        processor=framework_processor,
        inputs=preprocess_args.inputs,
        outputs=preprocess_args.outputs,
        job_arguments=preprocess_args.arguments,
        code=preprocess_args.code,
    )

    # Train step
    # Training steps have their own instance -> TrainingStep. However, you can use a
    # ProcessingStep without problems. Here we are going to see how it would be to use the own instance of training.
    training_parameters = {
        "entry_point": path_join(BASE_DIR, "code", "train.py"),
        "dependencies": [
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "train_utils.py"),
            path_join(BASE_DIR, "packages", "requirements", "train.txt"),
            path_join(BASE_DIR, "packages", "utils.py"),
        ],  # This argument is for passing needed files.
        "py_version": "py3",
        "role": role,
        "instance_count": 1,
        "instance_type": "ml.m5.4xlarge",  # Type of machine used in this step.
        "framework_version": "1.0-1",  # Framework version, in this case is the version of Scikit-learn.
        "base_job_name": f"{base_job_prefix}/sklearn_estimator",
        "container_log_level": logging.INFO,  # Important to see the logs on CloudWatch later.
        "hyperparameters": {
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "execution_date": execution_date,
            "is_for_predict": is_for_predict,
        },  # This argument is to pass the arguments to the executable.
    }
    sklearn_estimator = SKLearn(**training_parameters)
    training_step = TrainingStep(name="training_step", estimator=sklearn_estimator)

    # Predict step
    framework_processor = processors.framework()
    predict_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "predict.py"),
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "predict.txt"),
        ],  # This argument is for passing needed files.
        arguments=[
            "--s3_bucket", s3_bucket,
            "--s3_key", s3_key,
            "--execution_date", execution_date,
            "--is_for_predict", is_for_predict,
        ],  # This argument is to pass the arguments to the executable.
    )
    predict_step = ProcessingStep(
        name="predict_step",
        processor=framework_processor,
        inputs=predict_args.inputs,
        outputs=predict_args.outputs,
        job_arguments=predict_args.arguments,
        code=predict_args.code,
    )

    # Condition step
    condition_step = ConditionStep(
        name="condition_step",
        depends_on=["preprocess_step"],
        conditions=[train_condition],
        if_steps=[training_step],
        else_steps=[predict_step],
    )

    # PIPELINE DEFINITION
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            execution_date,
            s3_bucket,
            s3_key_data_source,
            s3_key,
        ],
        steps=[etl_step, preprocess_step, condition_step],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
