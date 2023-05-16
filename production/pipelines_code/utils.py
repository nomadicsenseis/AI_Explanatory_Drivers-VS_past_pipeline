"""
Utils for code pipeline definition of the BLV model
"""
from os.path import dirname, realpath
from typing import Dict, Optional

from boto3 import Session as b3Session
from sagemaker.processing import FrameworkProcessor, ScriptProcessor
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.spark.processing import PySparkProcessor
from yaml import safe_load


class Processors:
    """Class of processors used in the project."""

    def __init__(
        self,
        base_job_prefix: str,
        role: str,
        instance_count: int,
        instance_type: str,
        sagemaker_session: Session,
    ) -> None:
        """Class constructor."""
        self.base_job_prefix = base_job_prefix
        self.role = role
        self.instance_count = instance_count
        self.sagemaker_session = sagemaker_session
        self.instance_type = instance_type

    def pyspark(self) -> PySparkProcessor:
        """Pyspark processor."""
        return PySparkProcessor(
            base_job_name=f"{self.base_job_prefix}/pyspark_processor",
            framework_version="3.1",
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            max_runtime_in_seconds=10800,
        )

    def sklearn(self) -> SKLearnProcessor:
        """Sklearn processor."""
        return SKLearnProcessor(
            base_job_name=f"{self.base_job_prefix}/sklearn_processor",
            framework_version="0.23-1",
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            sagemaker_session=self.sagemaker_session,
        )

    def framework(self) -> FrameworkProcessor:
        """Framework processor"""
        return FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version="1.0-1",
            base_job_name=f"{self.base_job_prefix}/framework_processor",
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            sagemaker_session=self.sagemaker_session,
        )


def get_session(region: str, default_bucket: Optional[str]) -> Session:
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        sagemaker.session.Session instance
    """

    boto_session = b3Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    sess = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )
    return sess


def read_config_data() -> Dict:
    """Read the config.yml file asociated.

    The config.yml file asociated is the one in the same path.

    Returns
    -------
        Dictionary with the configuration of the process.
    """
    base_path = dirname(realpath(__file__))
    config_file_path = f"{base_path}/config.yml"
    with open(config_file_path) as conf_file:
        configuration = conf_file.read()
    return safe_load(configuration)
