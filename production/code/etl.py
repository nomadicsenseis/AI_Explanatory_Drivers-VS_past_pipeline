import argparse
import utils
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import logging
import json

SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())

STEP = "ETL"


# We define the Arguments class that inherits from the AbstractArguments abstract class.
class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        # We call the constructor of the parent class.
        super().__init__()

        # We create an ArgumentParser object that will contain all the necessary arguments for the script.
        parser = argparse.ArgumentParser(description=f"Inputs for the {STEP} step.")

        # We define the arguments that will be passed to the script.
        # "--s3_bucket": is the name of the S3 bucket where the data will be stored or from where it will be read.
        parser.add_argument("--s3_bucket", type=str)

        # "--s3_path_read": is the path in the S3 bucket from where the data will be read.
        parser.add_argument("--s3_path_read", type=str)

        # "--s3_path_write": is the path in the S3 bucket where the data will be written.
        parser.add_argument("--s3_path_write", type=str)

        # "--str_execution_date": is the execution date of the script.
        parser.add_argument("--str_execution_date", type=str)

        # "--use_type": specifies the type of use, it can be "predict" to predict or "train" to train the model.
        parser.add_argument("--use_type", type=str, choices=["predict", "train"])

        # Finally, we parse the arguments and store them in the self.args property of the class.
        self.args = parser.parse_args()


def create_spark_session() -> SparkSession:
    """
    Creates and configures a new Spark session.

    Returns:
        SparkSession: a configured Spark session.
    """
    # Create a new Spark session
    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()

    # Get all current Spark configurations
    configuration = spark.sparkContext.getConf().getAll()

    # Process each configuration
    for config_key, config_value in configuration:
        # If the configuration key is "spark.yarn.dist.files" and has a value (is not None)
        if config_key == "spark.yarn.dist.files" and config_value is not None:
            # Add the file to SparkContext
            spark.sparkContext.addFile(config_value)

        # Register the configuration in the SageMaker logs
        SAGEMAKER_LOGGER.info(
            "userlog: Pyspark configuration %s:%s", config_key, config_value
        )

    # Return the Spark session
    return spark


if __name__ == "__main__":
    """Main functionality of the script."""
    report_dict = {
        "hyperparam": {
            "eta": {
                "value": 0.6
            }
        }
    }
    # your report
    evaluation_path = "/opt/ml/processing/logs/logs.json"

    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    # Arguments
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()
    USE_TYPE = args.use_type
    S3_BUCKET = args.s3_bucket
    S3_PATH_READ = args.s3_path_read
    S3_PATH_WRITE = args.s3_path_write
    STR_EXECUTION_DATE = args.str_execution_date
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]

    # SparkSession and pyspark variables
    spark = create_spark_session()

    # Config file read
    config = utils.read_config_data(path=SparkFiles.get(filename="config.yml"))
    config_variables = config.get("VARIABLES")
    config_etl = config.get(STEP)

    # READ DATA SOURCE
    s3_dir = f's3://{S3_BUCKET}/{S3_PATH_READ}/titanic.csv'
    df = spark.read.csv(s3_dir, header='true')
    SAGEMAKER_LOGGER.info("userlog: Read date path %s.", s3_dir)

    # ETL Code

    # CODE FOR ETL
    len_df = df.count()
    SAGEMAKER_LOGGER.info(f"userlog: Dataframe length --> {str(len_df)}")

    # Save data
    save_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/{year}{month}{day}/titanic.csv"
    SAGEMAKER_LOGGER.info("userlog: Saving information for etl step in %s.", save_path)
    df.coalesce(1).write.option("header", "true").mode("overwrite").csv(save_path)
