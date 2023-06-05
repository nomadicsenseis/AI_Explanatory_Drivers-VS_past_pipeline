"""
This file is the CI/CD executable to generate or update the pipelines on each
commit/push of code depending on the branch. Main branch is for production
purpose and develop branch is for develop purpose.
"""
import argparse
import logging
from typing import Callable, Optional

import boto3
import sagemaker
import utils
from production.pipelines_code.pipeline import \
    get_pipeline as pipeline_definition


def get_arguments() -> argparse.Namespace:
    """Get input arguments to create sagemaker pipelines.

    Returns
    -------
        Namespace/dictionary with the arguments.
    """
    parser = argparse.ArgumentParser(
        description="CI/CD inputs to create sagemaker pipelines"
    )
    parser.add_argument(
        "--environment", type=str, help="Environment to deploy the results"
    )
    parser.add_argument(
        "--bucket", type=str, help="Bucket to deploy the results"
    )
    parser.add_argument(
        "--prefix", type=str, help="Prefix to deploy the results"
    )
    return parser.parse_args()


def generate_pipeline(
    pipeline_callable: Callable,
    pipe_name: str,
    base_job_prefix: str,
    role: Optional[str],
    region: str,
    default_bucket: str,
    default_bucket_prefix: str
) -> None:
    """Generate pipeline given in the pipeline_callable.

    Parameters
    ----------
        pipeline_callable: Pipeline to generate.
        pipe_name: Name to generate the pipeline.
        base_job_prefix: Job preffix of the steps in the pipeline.
        role: Aws role.
        region: Aws region.
        default_bucket: Aws default s3 bucket
        default_bucket_prefix: Aws default s3 key.
    """
    pipe_step = pipeline_callable(
        region=region,
        role=role,
        default_bucket=default_bucket,
        pipeline_name=pipe_name,
        base_job_prefix=base_job_prefix,
        default_bucket_prefix=default_bucket_prefix
    )
    pipe_step.upsert(role_arn=role)


def create_or_update_pipelines():
    """Create or update pipelines based on the code under the production folder."""
    logger = logging.getLogger("create_pipeline")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    args = get_arguments()
    env = args.environment
    bucket = args.bucket
    prefix = args.prefix
    environment = "prod" if env == "production" else "sbx"
    config = utils.read_config_data()
    model_version = config.get("MODEL_VERSION")
    vertical = config.get("VERTICAL")
    b3_session = boto3.Session()
    region = b3_session.region_name
    role = config.get("SAGEMAKER_ROLE")

    # GENERATE STEPS
    logger.info("userlog: Generating the pipeline definition...")
    pipeline_name = f"ibdata-{vertical}-{model_version.lower()}-{environment}-ppl"
    logger.info("userlog: Pipeline name: %s", pipeline_name)
    generate_pipeline(
        pipeline_callable=pipeline_definition,
        pipe_name=pipeline_name,
        base_job_prefix=pipeline_name,
        role=role,
        default_bucket=bucket,
        default_bucket_prefix=prefix,
        region=region
    )
    logger.info("userlog: Pipeline generated.")


if __name__ == "__main__":
    create_or_update_pipelines()
