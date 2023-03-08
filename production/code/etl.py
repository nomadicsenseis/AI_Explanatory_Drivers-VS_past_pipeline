"""
ETL Step for churn.
"""
import argparse
import datetime
import logging
from datetime import date
from typing import Dict

import numpy as np
import utils
from pyspark import SparkFiles
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())

STEP = "ETL"


class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        super().__init__()
        parser = argparse.ArgumentParser(description=f"Inputs for {STEP} step.")
        parser.add_argument("--s3_bucket", type=str)
        parser.add_argument("--s3_path_read", type=str)
        parser.add_argument("--s3_path_write", type=str)
        parser.add_argument("--str_execution_date", type=str)
        parser.add_argument("--is_last_date", type=str, default="1")
        parser.add_argument("--use_type", type=str, choices=["predict", "train"])
        self.args = parser.parse_args()


def create_spark_session() -> SparkSession:
    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    configuration = spark.sparkContext.getConf().getAll()
    for pys_conf in configuration:
        if (pys_conf[0] == "spark.yarn.dist.files") and (pys_conf[1] is not None):
            spark.sparkContext.addFile(pys_conf[1])
        SAGEMAKER_LOGGER.info(
            "userlog: Pyspark configuration %s:%s", pys_conf[0], pys_conf[1]
        )
    return spark


if __name__ == "__main__":
    """Main functionality of the script."""

    # Arguments
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()
    USE_TYPE = args.use_type
    S3_BUCKET = args.s3_bucket
    S3_PATH_READ = args.s3_path_read
    S3_PATH_WRITE = args.s3_path_write
    # SparkSession and pyspark variables
    spark = create_spark_session()
    # Config file read
    config = utils.read_config_data(path=SparkFiles.get(filename="config.yml"))
    config_variables = config.get("VARIABLES")
    config_etl = config.get(STEP)
    # READ DATA
    s3_dir = ''
    df = spark.read.csv(s3_dir, header='true')
    SAGEMAKER_LOGGER.info("userlog: Read date path %s.", s3_dir)
    # ETL Code
    # CODE FOR ETL
    len_df = df.count()
    SAGEMAKER_LOGGER.info(f"userlog: Dataframe length --> {str(len_df)}")
    # Save data
    save_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/"
    SAGEMAKER_LOGGER.info("userlog: Saving information for etl step in %s.", save_path)
    df.write.option("header", "true").mode("overwrite").csv(save_path)
