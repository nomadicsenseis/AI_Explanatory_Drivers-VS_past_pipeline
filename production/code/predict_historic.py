from subprocess import check_call
from sys import executable

STEP = "PREDICT_HISTORIC"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

# General
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from datetime import datetime, timedelta
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import os
import numpy as np
import xlsxwriter
import datetime
import boto3
import s3fs
from itertools import combinations
import pickle
import json
import re
import gc
import argparse
import logging
from os import environ
import utils
from boto3 import resource
from pandas import read_csv

# Sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Models
from catboost import CatBoostClassifier, cv, Pool
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP
import shap

# Random
import random

#Warnings
import warnings
warnings.filterwarnings("ignore")

SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())


# Inherits from the AbstractArguments class
class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        # Call to the constructor of the parent class
        super().__init__()

        # Create an ArgumentParser object
        parser = argparse.ArgumentParser(description=f"Inputs for {STEP} step.")

        # Add the command line arguments that will be used
        parser.add_argument("--s3_bucket", type=str)  # S3 bucket name
        parser.add_argument("--s3_path_write", type=str)  # S3 path to write data
        parser.add_argument("--str_execution_date", type=str)  # Execution date
        parser.add_argument("--is_last_date", type=str, default="1")  # Indicator for the last date

        # Parse the arguments and store them in the 'args' attribute
        self.args = parser.parse_args()


if __name__ == "__main__":
    """Main functionality of the script."""

    # Log the start of the step
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)

    # Initialize the Arguments class and get the arguments
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()

    # Extract the argument values
    S3_BUCKET = args.s3_bucket
    S3_PATH_WRITE = args.s3_path_write
    STR_EXECUTION_DATE = args.str_execution_date
    IS_LAST_DATE = args.is_last_date

    # Parse date from STR_EXECUTION_DATE
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]

    # Load the configuration data
    config = utils.read_config_data()
    model_names = list(config['TRAIN']['MODEL_NAME'])

    # Define the paths for reading data and the trained model
    read_path = f"{S3_PATH_WRITE}/01_preprocess_step/train/{year}{month}{day}"
    clf_model={}
    for name in model_names:
        path_read_train = f"{S3_PATH_WRITE}/02_train_step/{name}"

        # Determine the path to read the model from
        model_path, _, _, _ = utils.get_path_to_read_and_date(
            read_last_date=bool(int(IS_LAST_DATE)),
            bucket=S3_BUCKET,
            key=path_read_train,
            partition_date=STR_EXECUTION_DATE,
        )

        # Extract the bucket and object key from the model_path
        if 's3://' in model_path:
            model_path = model_path.split('//')[1].replace(f"{S3_BUCKET}/", '')
        SAGEMAKER_LOGGER.info(f"userlog: path for models: {model_path + '/model/'}")

        # Load the trained model from S3
        s3_resource = resource("s3")
        fitted_clf_model = (
            s3_resource.Bucket(S3_BUCKET)
            .Object(f"{model_path}/model/{config['TRAIN']['MODEL_NAME']}")
            .get()
        )
        clf_model[name] = pickle.loads(fitted_clf_model["Body"].read())

    # Load the data to predict
    df_predict = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_for_historic_prediction.csv")

    # Perform prediction and add the probabilities to the dataframe
    features = list(config['TRAIN']['FEATURES'])
    missing_rows = df_predict[features].isnull().any(axis=1)
    df_predict = df_predict[~missing_rows]
    df_probabilities = utils.predict_and_explain(clf_model[model_names[0]], clf_model[model_names[1]], df_predict, features)

    # Rename columns, add insert date and select columns to save
    df_probabilities['insert_date_ci'] = STR_EXECUTION_DATE
    df_probabilities = df_probabilities[config['PREDICT']['COLUMNS_SAVE']]

    # Save the prediction results to S3
    save_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/04_predict_historic_step/{year}{month}{day}/historic_predictions.csv"
    SAGEMAKER_LOGGER.info("userlog: Saving information for predict step in %s.", save_path)
    df_predict.to_csv(save_path, index=False)

