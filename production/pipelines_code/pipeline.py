import logging
import sagemaker
from os import chdir, pardir, sep
from os.path import dirname
from os.path import join as path_join
from os.path import realpath
from typing import Optional

from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep
from production.pipelines_code import utils
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingOutput
from sagemaker.session import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.estimator import Estimator
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

BASE_DIR = dirname(realpath(f"{__file__}{sep}{pardir}"))
chdir(BASE_DIR)

MIN_ROW_NUMBER = "160000000"

def get_pipeline(
    region: str,  # AWS region
    pipeline_name: str,  # Name of the pipeline
    base_job_prefix: str,  # Prefix for the job names
    role: Optional[str] = None,  # IAM role
    default_bucket: Optional[str] = None,  # Default S3 bucket
    default_bucket_prefix: Optional[str] = None,  # Default S3 key
) -> Pipeline:

    # Get a Sagemaker session
    sagemaker_session = utils.get_session(region=region, default_bucket=default_bucket,
                                          default_bucket_prefix=default_bucket_prefix)

    # If the role is not provided, get the execution role
    if role is None:
        role = get_execution_role(sagemaker_session)

    # Define pipeline parameters
    processing_instance_count = ParameterInteger(name="processing_instance_count", default_value=3)
    param_str_execution_date = ParameterString(name="str_execution_date", default_value="2023-03-01")
    param_s3_bucket_nps = ParameterString(name="s3_bucket_nps", default_value="iberia-data-lake")
    param_s3_bucket_lf = ParameterString(name="s3_bucket_lf", default_value="ibdata-prod-ew1-s3-customer")
    param_s3_path_read_nps = ParameterString(name="s3_path_read_nps")
    param_s3_path_read_lf = ParameterString(name="s3_path_read_lf")
    param_s3_path_write = ParameterString(name="s3_path_write")
    param_is_last_date = ParameterString(name="is_last_date", default_value="1")
    param_use_type = ParameterString(name="use_type")
    param_trials = ParameterString(name="trials", default_value="1")
    param_is_retrain_required = ParameterString(name="is_retrain_required", default_value="1")

    # Define a condition for pipeline execution
    train_predict_condition = ConditionEquals(left=param_use_type, right="train")

    # Read the configuration file
    configuration = utils.read_config_data()

    # Initialize the processors used in the pipeline executions
    processors = utils.Processors(
        base_job_prefix=base_job_prefix,
        role=role,
        instance_count=processing_instance_count,
        instance_type="ml.m5.4xlarge",
        sagemaker_session=sagemaker_session,
    )

    # ETL
    # Initialize the PySpark processor
    framework_processor = processors.framework()

    # Define the arguments for running a PySpark job
    etl_step_args = framework_processor.get_run_args(
        # Path to the preprocessing script
        code=path_join(BASE_DIR, "code", "etl.py"),

        # List of dependencies required by the preprocessing script
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "etl.txt")
        ],
        arguments=[  # Command line arguments to the framework script
            "--s3_bucket_nps",
            param_s3_bucket_nps,
            "--s3_bucket_lf",
            param_s3_bucket_lf,
            "--s3_path_read_nps",
            param_s3_path_read_nps,
            "--s3_path_read_lf",
            param_s3_path_read_lf,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--use_type",
            param_use_type,
        ],
    )

    # Create a processing step
    etl_step = ProcessingStep(
        name="etl_step",  # Name of the step
        processor=framework_processor,  # Processor to be used (framework in this case)
        inputs=etl_step_args.inputs,  # Inputs for the processor
        outputs=etl_step_args.outputs,  # Where to store the outputs
        job_arguments=etl_step_args.arguments,  # Arguments for the processor
        code=etl_step_args.code,  # Code to be executed
    )

    # PREPROCESS
    # Instantiate the processor
    framework_processor = processors.framework()

    # Configure the arguments for the processing step
    train_preprocess_step_args = framework_processor.get_run_args(
        # Path to the preprocessing script
        code=path_join(BASE_DIR, "code", "preprocess.py"),

        # List of dependencies required by the preprocessing script
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "preprocess.txt")
        ],

        # Arguments to pass to the preprocessing script
        arguments=[
            "--s3_bucket",
            param_s3_bucket_nps,  # S3 bucket for data storage
            "--s3_path_write",
            param_s3_path_write,  # Path for writing outputs
            "--str_execution_date",
            param_str_execution_date,  # Execution date string
            "--use_type",
            param_use_type,  # Use type parameter
        ],
    )

    # Define the processing step
    preprocess_step = ProcessingStep(
        # Step name
        name="preprocess_step",# Processor to use
        depends_on=["etl_step"],
        processor=framework_processor,# Input data
        inputs=train_preprocess_step_args.inputs,# Output configuration
        outputs=train_preprocess_step_args.outputs,# Arguments for the job
        job_arguments=train_preprocess_step_args.arguments,# Code to execute
        code=train_preprocess_step_args.code,
    )

    
    # TRAIN
    # Create a framework processor for training
    framework_processor = processors.framework()

    # Generate the arguments for the training step
    train_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "train.py"),  # Specify the path to the training code
        dependencies=[
            path_join(BASE_DIR, "packages", "config.yml"),  # Specify the dependencies required for training
            path_join(BASE_DIR, "packages", "requirements", "train.txt"),
            path_join(BASE_DIR, "packages", "utils.py"),
        ],
        arguments=[
            "--s3_bucket", param_s3_bucket_nps,  # Pass the S3 bucket as an argument
            "--s3_path_write", param_s3_path_write,  # Pass the S3 path for writing as an argument
            "--str_execution_date", param_str_execution_date,  # Pass the execution date as an argument
            "--is_last_date", param_is_last_date,  # Pass the flag for last date as an argument
        ],
    )

    # Define the training step in the pipeline
    train_step = ProcessingStep(
        name="train_step",  # Set a name for the step
        # Specify that this step depends on the completion of the "train_preprocess_step"
        processor=framework_processor,  # Use the framework processor
        inputs=train_step_args.inputs,  # Specify the inputs for this step
        outputs=train_step_args.outputs,  # Specify the outputs for this step
        job_arguments=train_step_args.arguments,  # Specify the job arguments for this step
        code=train_step_args.code,  # Provide the code for this step
    )

    # PREDICT
    framework_processor = processors.framework()
    predict_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "predict.py"),  # Path to the predict.py script
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),  # Path to the utils.py file
            path_join(BASE_DIR, "packages", "config.yml"),  # Path to the config.yml file
            path_join(BASE_DIR, "packages", "requirements", "predict.txt"),  # Path to the predict.txt file
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket_nps,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--is_last_date",
            param_is_last_date,
        ],
    )

    #PREDICT STEP
    predict_step = ProcessingStep(
        name="predict_step", # Depends on the previous step "predict_preprocess_step"
        processor=framework_processor,  # Processor to use for the step
        inputs=predict_step_args.inputs,  # Input data for the step
        outputs=predict_step_args.outputs,  # Output data for the step
        job_arguments=predict_step_args.arguments,  # Additional job arguments
        code=predict_step_args.code,  # Code to execute for the step
    )

    # PREDICT HISTORIC
    framework_processor = processors.framework()
    predict_historic_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "predict_historic.py"),  # Path to the predict.py script
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),  # Path to the utils.py file
            path_join(BASE_DIR, "packages", "config.yml"),  # Path to the config.yml file
            path_join(BASE_DIR, "packages", "requirements", "predict_historic.txt"),  # Path to the predict_historic.txt file
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket_nps,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--is_last_date",
            param_is_last_date,
        ],
    )

    #PREDICT HISTORIC STEP
    predict_historic_step = ProcessingStep(
        name="predict_historic_step",
        depends_on=["train_step"],  # Depends on the previous step "train_step"
        processor=framework_processor,  # Processor to use for the step
        inputs=predict_historic_step_args.inputs,  # Input data for the step
        outputs=predict_historic_step_args.outputs,  # Output data for the step
        job_arguments=predict_historic_step_args.arguments,  # Additional job arguments
        code=predict_historic_step_args.code,  # Code to execute for the step
    )

    # CONDITION STEP
    condition_step = ConditionStep(
        name="condition_step",
        depends_on=["preprocess_step"],  # Depends on the previous step "etl_step"
        conditions=[train_predict_condition],  # Condition for branching
        if_steps=[train_step, predict_historic_step],  # Steps to execute if the condition is true
        else_steps=[predict_step]  # Steps to execute if the condition is false
    )

    # PIPELINE
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,  # Number of instances for data processing
            param_str_execution_date,  # Execution date as a string
            param_s3_bucket_nps,  # S3 bucket name
            param_s3_bucket_lf,  # S3 bucket name
            param_s3_path_read_nps,  # S3 path for reading the nps data
            param_s3_path_read_lf,  # S3 path for reading the lf data
            param_s3_path_write,  # S3 path for writing data
            param_is_last_date,  # Flag indicating if it is the last date for data processing
            param_use_type,  # Type of data to be processed
            param_trials,  # Number of trials for model training
            param_is_retrain_required,  # Flag indicating if retraining is required
        ],
        steps=[
               etl_step,
               preprocess_step, 
               condition_step
              ],  # List of pipeline steps
        sagemaker_session=sagemaker_session,  # Sagemaker session object
    )
    return pipeline
