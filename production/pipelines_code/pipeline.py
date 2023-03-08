"""
Pipeline definition for BLV model.
"""
import logging
import sagemaker
from os import chdir, pardir, sep
from os.path import dirname
from os.path import join as path_join
from os.path import realpath
from typing import Optional

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
    region: str,
    pipeline_name: str,
    base_job_prefix: str,
    role: Optional[str] = None,
    default_bucket: Optional[str] = None,
) -> Pipeline:
    """Pipeline definition."""
    sagemaker_session = utils.get_session(region=region, default_bucket=default_bucket)
    if role is None:
        role = get_execution_role(sagemaker_session)
    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="processing_instance_count", default_value=3)
    param_str_execution_date = ParameterString(name="str_execution_date", default_value="2023-03-01")
    param_s3_bucket = ParameterString(name="s3_bucket",default_value="iberia-data-lake")
    param_s3_path_read = ParameterString(name="s3_path_read")
    param_s3_path_write = ParameterString(name="s3_path_write")
    param_is_last_date = ParameterString(name="is_last_date", default_value="1")
    param_use_type = ParameterString(name="use_type")
    param_trials = ParameterString(name="trials", default_value="1")
    param_is_retrain_required = ParameterString(name="is_retrain_required", default_value="1")
    train_predict_condition = ConditionEquals(left=param_use_type, right="train")
    # -------------------------------------------
    configuration = utils.read_config_data()
    pyspark_properties = {
        key.replace("_", "."): value
        for key, value in configuration.get("PYSPARK").items()
    }
    # Pyspark Configuration
    pyspark_config = [
        {"Classification": "spark-defaults", "Properties": pyspark_properties}
    ]
    # Procesors used in the executions
    processors = utils.Processors(
        base_job_prefix=base_job_prefix,
        role=role,
        instance_count=processing_instance_count,
        instance_type="ml.m5.4xlarge",
        sagemaker_session=sagemaker_session,
    )
    # train_processors = utils.Processors(
    #     base_job_prefix=base_job_prefix,
    #     role=role,
    #     instance_type="ml.m5.4xlarge",
    #     instance_count=processing_instance_count,
    #     sagemaker_session=sagemaker_session,
    # )

    # ETL ERRORS
    pyspark_processor = processors.pyspark()
    etl_errors_step_pyspark_args = pyspark_processor.get_run_args(
        submit_app=path_join(BASE_DIR, "code", "etl_errors.py"),
        submit_py_files=[
            path_join(BASE_DIR, "packages", "utils.py"),
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket,
            "--s3_path_read",
            param_s3_path_read,
            "--str_execution_date",
            param_str_execution_date,
            "--is_last_date",
            param_is_last_date,
            "--min_row_number",
            MIN_ROW_NUMBER
        ],
        configuration=pyspark_config,
    )
    etl_errors_step = ProcessingStep(
        name="etl_errors_step",
        processor=pyspark_processor,
        inputs=etl_errors_step_pyspark_args.inputs,
        outputs=etl_errors_step_pyspark_args.outputs,
        job_arguments=etl_errors_step_pyspark_args.arguments,
        code=etl_errors_step_pyspark_args.code,
    )

    # ETL
    pyspark_processor = processors.pyspark()
    etl_step_pyspark_args = pyspark_processor.get_run_args(
        submit_app=path_join(BASE_DIR, "code", "etl.py"),
        submit_py_files=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "etl_utils.py"),
        ],
        submit_files=[path_join(BASE_DIR, "packages", "config.yml")],
        arguments=[
            "--s3_bucket",
            param_s3_bucket,
            "--s3_path_read",
            param_s3_path_read,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--is_last_date",
            param_is_last_date,
            "--use_type",
            param_use_type,
        ],
        configuration=pyspark_config,
    )
    etl_step = ProcessingStep(
        name="etl_step",
        depends_on=["etl_errors_step"],
        processor=pyspark_processor,
        inputs=etl_step_pyspark_args.inputs,
        outputs=etl_step_pyspark_args.outputs,
        job_arguments=etl_step_pyspark_args.arguments,
        code=etl_step_pyspark_args.code,
    )

    # PREPROCESS TRAIN
    framework_processor = processors.framework()
    train_preprocess_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "preprocess.py"),
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "preprocess.txt")
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--use_type",
            param_use_type,
        ],
    )
    train_preprocess_step = ProcessingStep(
        name="train_preprocess_step",
        processor=framework_processor,
        inputs=train_preprocess_step_args.inputs,
        outputs=train_preprocess_step_args.outputs,
        job_arguments=train_preprocess_step_args.arguments,
        code=train_preprocess_step_args.code,
    )

    # PREPROCESS PREDICT
    framework_processor = processors.framework()
    predict_preprocess_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "preprocess.py"),
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "preprocess.txt"),
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--use_type",
            param_use_type,
            "--is_last_date",
            param_is_last_date,
        ],
    )
    predict_preprocess_step = ProcessingStep(
        name="predict_preprocess_step",
        processor=framework_processor,
        inputs=predict_preprocess_step_args.inputs,
        outputs=predict_preprocess_step_args.outputs,
        job_arguments=predict_preprocess_step_args.arguments,
        code=predict_preprocess_step_args.code,
    )

    # TRAIN AUX
    framework_processor = processors.framework()
    train_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "train.py"),
        dependencies=[
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "train_utils.py"),
            path_join(BASE_DIR, "packages", "requirements", "train.txt"),
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "plots.py"),
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--is_last_date",
            param_is_last_date,
        ],
    )
    train_step = ProcessingStep(
        name="train_step",
        depends_on=["train_preprocess_step"],
        processor=framework_processor,
        inputs=train_step_args.inputs,
        outputs=train_step_args.outputs,
        job_arguments=train_step_args.arguments,
        code=train_step_args.code,
    )

    # TRAIN
    training_estimator_parameters = {
        "entry_point": path_join(BASE_DIR, "code", "train.py"),
        "dependencies": [
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "train_utils.py"),
            path_join(BASE_DIR, "packages", "requirements", "train.txt"),
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "plots.py"),
        ],
        "py_version": "py3",
        "role": role,
        "instance_count": 1,
        "instance_type": "ml.m5.4xlarge",
        "framework_version": "1.0-1",
        "base_job_name": f"{base_job_prefix}/sklearn_estimator",
        "container_log_level": logging.INFO,
        "hyperparameters": {
            "s3_bucket": param_s3_bucket,
            "s3_path_write": param_s3_path_write,
            "str_execution_date": param_str_execution_date,
            "is_last_date": param_is_last_date,
        },
    }
    sklearn_estimator = SKLearn(**training_estimator_parameters)
    training_step = TrainingStep(
        name="training_step", estimator=sklearn_estimator# , depends_on=["train_preprocess_step"]
    )
    #
    # # EVALUATION
    # train_step_evaluation_report = PropertyFile(
    #     name="train_step_evaluation_report",
    #     output_name="train_step_evaluation_report",
    #     path="train_step_evaluation_report.json",
    # )
    # framework_processor = processors.framework()
    # evaluation_step_args = framework_processor.get_run_args(
    #     code=path_join(BASE_DIR, "code", "evaluation.py"),
    #     dependencies=[
    #         path_join(BASE_DIR, "packages", "utils.py"),
    #         path_join(BASE_DIR, "packages", "config.yml"),
    #         path_join(BASE_DIR, "packages", "requirements", "evaluation.txt"),
    #     ],
    #     arguments=[
    #         "--s3_bucket",
    #         param_s3_bucket,
    #         "--s3_path",
    #         param_s3_path,
    #         "--str_execution_date",
    #         param_str_execution_date,
    #         "--is_last_date",
    #         param_is_last_date,
    #         "--model_type",
    #         param_model_type,
    #         "--is_retrain_required",
    #         param_is_retrain_required
    #     ],
    #     outputs=[
    #         ProcessingOutput(
    #             output_name="train_step_evaluation_report",
    #             source="/opt/ml/processing/train_step_evaluation_report",
    #         )
    #     ],
    # )
    # evaluation_step = ProcessingStep(
    #     name="evaluation_step",
    #     depends_on=["training_step"],
    #     processor=framework_processor,
    #     inputs=evaluation_step_args.inputs,
    #     outputs=evaluation_step_args.outputs,
    #     job_arguments=evaluation_step_args.arguments,
    #     code=evaluation_step_args.code,
    #     property_files=[train_step_evaluation_report],
    # )
    # model_metrics = ModelMetrics(
    #     model_statistics=MetricsSource(
    #         content_type="application/json",
    #         s3_uri=f"{evaluation_step.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']}/train_step_evaluation_report.json",
    #     )
    # )
    # register_step = RegisterModel(
    #     name="register_step",
    #     estimator=sklearn_estimator,
    #     model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    #     depends_on=["evaluation_step"],
    #     response_types=["text/csv"],
    #     content_types=["text/csv"],
    #     approval_status="Approved",
    #     model_metrics=model_metrics,
    #     domain="MACHINE_LEARNING",
    #     task="CLASSIFICATION",
    #     model_package_group_name=f"{pipeline_name}-package-group",
    # )
    # PREDICT
    framework_processor = processors.framework()
    predict_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "predict.py"),
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "predict.txt"),
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--is_last_date",
            param_is_last_date,
        ],
    )
    predict_step = ProcessingStep(
        name="predict_step",
        depends_on=["predict_preprocess_step"],
        processor=framework_processor,
        inputs=predict_step_args.inputs,
        outputs=predict_step_args.outputs,
        job_arguments=predict_step_args.arguments,
        code=predict_step_args.code,
    )

    # EXPLAINER
    framework_processor = processors.framework()
    explainer_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "explainer.py"),
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "explainer.txt"),
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--is_last_date",
            param_is_last_date,
        ],
    )
    explainer_step = ProcessingStep(
        name="explainer_step",
        depends_on=["train_step"],
        processor=framework_processor,
        inputs=explainer_step_args.inputs,
        outputs=explainer_step_args.outputs,
        job_arguments=explainer_step_args.arguments,
        code=explainer_step_args.code,
    )

    # CONDITION STEP
    condition_step = ConditionStep(
        name="condition_step",
        depends_on=["etl_step"],
        conditions=[train_predict_condition],
        if_steps=[train_preprocess_step, train_step, explainer_step],
        else_steps=[predict_preprocess_step, predict_step],
    )
    # PIPELINE
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            param_str_execution_date,
            param_s3_bucket,
            param_s3_path_read,
            param_s3_path_write,
            param_is_last_date,
            param_use_type,
            param_trials,
            param_is_retrain_required,
        ],
        steps=[etl_step, condition_step],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
