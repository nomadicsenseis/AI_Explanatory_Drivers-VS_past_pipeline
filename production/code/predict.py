"""
Predict Step for churn.
"""
from subprocess import check_call
from sys import executable

STEP = "PREDICT"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

import argparse
import logging
import numpy as np
from pickle import loads as pkl_loads

import utils
from boto3 import resource
from pandas import read_csv

SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())

class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        super().__init__()
        parser = argparse.ArgumentParser(description=f"Inputs for {STEP} step.")
        parser.add_argument("--s3_bucket", type=str)
        parser.add_argument("--s3_path_write", type=str)
        parser.add_argument("--str_execution_date", type=str)
        parser.add_argument("--is_last_date", type=str, default="1")
        self.args = parser.parse_args()


if __name__ == "__main__":
    """Main functionality of the script."""

    # DEFINE ARGUMENTS
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()
    S3_BUCKET = args.s3_bucket
    S3_PATH_WRITE = args.s3_path_write
    STR_EXECUTION_DATE = args.str_execution_date
    IS_LAST_DATE = args.is_last_date
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]
    config = utils.read_config_data()
    # Paths
    read_path = f"{S3_PATH_WRITE}/01_preprocess_step/predict/{year}{month}{day}"
    path_read_train = f"{S3_PATH_WRITE}/02_train_step"
    # Path for reading model
    model_path, _, _, _ = utils.get_path_to_read_and_date(
        read_last_date=bool(int(IS_LAST_DATE)),
        bucket=S3_BUCKET,
        key=path_read_train,
        partition_date=STR_EXECUTION_DATE,
    )
    if 's3://' in model_path:
        model_path = model_path.split('//')[1].replace(f"{S3_BUCKET}/", '')
    SAGEMAKER_LOGGER.info(f"userlog: path for models: {model_path + '/model/'}")
    s3_resource = resource("s3")
    fitted_clf_model = (
        s3_resource.Bucket(S3_BUCKET)
        .Object(f"{model_path}/model/{config['TRAIN']['MODEL_NAME_SAVE']}")
        .get()
    )
    clf_model = pkl_loads(fitted_clf_model["Body"].read())
    # READ DATA | PREDICT
    df_predict = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_for_prediction.csv")
    # Predict proba
    probabilities = clf_model.predict_proba(
        df_predict[clf_model.feature_names_]
    )
    df_predict[f"{config['VARIABLES']['TARGET']}_probability"] = probabilities[:, 1]
    df_predict = df_predict.rename(config['PREDICT']['COLUMNS_RENAME'], axis=1)
    df_predict['insert_date_ci'] = STR_EXECUTION_DATE
    df_predict = df_predict[config['PREDICT']['COLUMNS_SAVE']]
    # SAVE DATA
    save_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/03_predict_step/{year}{month}{day}/predictions.csv"
    SAGEMAKER_LOGGER.info("userlog: Saving information for predict step in %s.", save_path)
    df_predict.to_csv(save_path, index=False)
    # Log churn info
    churn = np.round((len(df_predict[df_predict['churn_probability'] >= 0.5])/len(df_predict))*100, 2)
    SAGEMAKER_LOGGER.info(f"userlog: Churners percentage --> {str(churn)}%", save_path)
