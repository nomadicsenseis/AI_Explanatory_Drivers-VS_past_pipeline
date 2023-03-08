"""
PREPROCESS Step for churn.
"""
from subprocess import check_call
from sys import executable
from typing import Any, Dict

STEP = "PREPROCESS"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

import argparse
import logging
import pandas as pd
import pickle
import time
from pandas import DataFrame

import boto3
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer

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
        parser.add_argument("--use_type", type=str, choices=["predict", "train"])

        self.args = parser.parse_args()


def read_data_churn() -> DataFrame:
    """This function automatically reads a dataframe processed
    with all features in S3 and return this dataframe with
    cid as index

    Parameters
    ----------

    Returns
    -------
    Pandas dataframe containing all features
    """

    s3_keys = [item.key for item in s3_resource.Bucket(S3_BUCKET).objects.filter(Prefix=prefix) if item.key.endswith(".csv")]
    preprocess_paths = [f"s3://{S3_BUCKET}/{key}" for key in s3_keys]
    df_features_churn = pd.DataFrame()
    for file in preprocess_paths:
        df = pd.read_csv(file)
        df_features_churn = pd.concat([df_features_churn, df], axis=0)
    df_features_churn.index = df_features_churn[config['VARIABLES']['ID']]
    df_features_churn.index.name = config['VARIABLES']['ID']
    SAGEMAKER_LOGGER.info(f"Data size: {str(len(df_features_churn))}")
    SAGEMAKER_LOGGER.info(f"Columns: {df_features_churn.columns}")
    return df_features_churn


def extract_trend(df: DataFrame) -> DataFrame:
    """This function creates two new columns from a global lisit
    of yearly variables. It computes the trends between year 1 and 2
    and between year 2 and 3

    Parameters
    ----------
    df: Pandas dataframe with features

    Returns
    -------
    Pandas dataframe with new columns containing trends between columns
    """

    for col in config['PREPROCESS']['COLUMNS_TREND']:
        df['trend1_' + col] = ((df[col + '_year1'] - df[col + '_year2']) / df[col + '_year2']).replace(np.inf,
                                                                                                       1).replace(
            -np.inf, 1).replace(np.nan, 0)
        df['trend2_' + col] = ((df[col + '_year2'] - df[col + '_year3']) / df[col + '_year3']).replace(np.inf,
                                                                                                       1).replace(
            -np.inf, 1).replace(np.nan, 0)

    return df


def auxiliar_type_conversion(df: DataFrame) -> DataFrame:
    """This function converts types from feature dataframe

    Parameters
    ----------
    df: Pandas dataframe with features

    Returns
    -------
    Pandas dataframe with casted columns
    """

    # Categorical type
    for col in config['PREPROCESS']['CAT_CONV']:
        df[col] = df[col].astype('category')
    # Float type
    for col in config['PREPROCESS']['FLOAT_CONV']:
        df[col] = df[col].astype(float)

    return df


def feature_encoder_train(X_train: DataFrame, X_test: DataFrame, X_val: DataFrame) -> DataFrame:
    # Isolate target variable
    train_target = X_train[config['VARIABLES']['TARGET']]
    X_train = X_train.drop(config['VARIABLES']['TARGET'], axis=1)
    test_target = X_test[config['VARIABLES']['TARGET']]
    X_test = X_test.drop(config['VARIABLES']['TARGET'], axis=1)
    val_target = X_val[config['VARIABLES']['TARGET']]
    X_val = X_val.drop(config['VARIABLES']['TARGET'], axis=1)
    # Binning age
    Kbin_age = KBinsDiscretizer(n_bins=config['PREPROCESS']['BIN_AGE'], encode='ordinal', strategy='quantile')
    trf_cat = ColumnTransformer([('age_bin', Kbin_age, ['age'])])
    X_train.loc[X_train['age'].notnull(), 'age_bin'] = trf_cat.fit_transform(X_train[X_train['age'].notnull()])
    X_test.loc[X_test['age'].notnull(), 'age_bin'] = trf_cat.transform(X_test[X_test['age'].notnull()])
    X_val.loc[X_val['age'].notnull(), 'age_bin'] = trf_cat.transform(X_val[X_val['age'].notnull()])
    # Target encoder
    tg_encoder = TargetEncoder()
    X_train['most_od_flown_tg'] = tg_encoder.fit_transform(X_train['most_od_flown'], train_target)
    X_test['most_od_flown_tg'] = tg_encoder.transform(X_test['most_od_flown'])
    X_val['most_od_flown_tg'] = tg_encoder.transform(X_val['most_od_flown'])
    # Save binning encoder
    trf_cat_dump = pickle.dumps(trf_cat)
    s3_resource.Object(S3_BUCKET, f"{save_path}/models/{config['PREPROCESS']['CAT_ENCODER']}").put(
        Body=trf_cat_dump
    )
    # Save binning encoder
    tg_encoder_dump = pickle.dumps(tg_encoder)
    s3_resource.Object(S3_BUCKET, f"{save_path}/models/{config['PREPROCESS']['TARGET_ENCODER']}").put(
        Body=tg_encoder_dump
    )
    # Add target column
    X_train[config['VARIABLES']['TARGET']] = train_target
    X_test[config['VARIABLES']['TARGET']] = test_target
    X_val[config['VARIABLES']['TARGET']] = val_target

    return X_train, X_test, X_val


def feature_encoder_predict(X: DataFrame) -> DataFrame:

    # Path for reading
    model_path, _, _, _ = utils.get_path_to_read_and_date(
        read_last_date=bool(int(IS_LAST_DATE)),
        bucket=S3_BUCKET,
        key=path_read,
        partition_date=STR_EXECUTION_DATE,
    )
    # Eliminate s3 info path
    if 's3://' in model_path:
        model_path = model_path.split('//')[1].replace(f"{S3_BUCKET}/", '')
    SAGEMAKER_LOGGER.info(f"userlog: path for models: {model_path + '/models/'}")
    # Read binning encoder
    trf_cat = (
        s3_resource.Object(S3_BUCKET, f"{model_path}/models/{config['PREPROCESS']['CAT_ENCODER']}")
        .get()
    )
    trf_cat = pickle.loads(trf_cat["Body"].read())
    time.sleep(30)
    # Read target encoder
    tg_encoder = (
        s3_resource.Object(S3_BUCKET, f"{model_path}/models/{config['PREPROCESS']['TARGET_ENCODER']}")
        .get()
    )
    tg_encoder = pickle.loads(tg_encoder["Body"].read())

    # Impute new data
    X.loc[X['age'].notnull(), 'age_bin'] = trf_cat.transform(X[X['age'].notnull()])
    X['most_od_flown_tg'] = tg_encoder.transform(X['most_od_flown'])

    return X
def read_data_and_fe_process() -> object:
    """This reads churn daa from s3 (last partition), creates some features
    and preprocess some columns in order to finally divide this dataframe
    between churn and cltv (with no churners)

    Parameters
    ----------
    None

    Returns
    -------
    Two dataframes containing all data preprocess (churn) and cltv data
    preprocess (only not churners)
    """

    # Read data
    df = read_data_churn()

    # Extract trend
    df = extract_trend(df)

    # Last purchase new features
    df['last_purchase1_2'] = df['last2_purchase'] - df['last_purchase']
    df['last_purchase2_3'] = df['last3_purchase'] - df['last2_purchase']

    # Cast variables
    df = auxiliar_type_conversion(df)

    return df


def split_train_val_test(X: DataFrame) -> object:
    """This function uses the pandas dataframe features, divide between
    target and features and splits this data between:
    - X_train_churn, y_train_churn: training for churn
    - X_test_churn, y_test_churn: testing for churn
    - X_final_test, y_final_test: final test for churn and cltv (unseen data)

    Parameters
    ----------
    df: pandas dataframe with churn features and label

    Returns
    -------
    All dataframes for training and testing, a set of data randomly sampled
    from churn to introduce to cltv model (in order to have some zero
    predictions) and those ids selected for this purpose
    """

    X = X[features_target_fin]
    # Split between train, validation and test
    X_train, X_test = train_test_split(X, test_size=config['PREPROCESS']['TEST_SIZE'],
                                       random_state=1)
    X_test, X_val = train_test_split(X_test, test_size=config['PREPROCESS']['VAL_TEST_SIZE'],
                                     random_state=1)

    return X_train, X_test, X_val

if __name__ == "__main__":

    """Main functionality of the script."""
    # DEFINE ARGUMENTS
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()
    config = utils.read_config_data()
    S3_BUCKET = args.s3_bucket
    S3_PATH_WRITE = args.s3_path_write
    USE_TYPE = args.use_type
    STR_EXECUTION_DATE = args.str_execution_date
    IS_LAST_DATE = args.is_last_date
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]
    # s3 object
    s3_resource = boto3.resource("s3")
    # path
    prefix = f"{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/{STR_EXECUTION_DATE.replace('-', '')}"
    save_path = f"{S3_PATH_WRITE}/01_preprocess_step/{USE_TYPE}/{year}{month}{day}"
    save_path_train = f"{S3_PATH_WRITE}/01_preprocess_step/train/{year}{month}{day}"
    path_read = f"{S3_PATH_WRITE}/01_preprocess_step/train"
    # Read data
    df_features_churn = read_data_and_fe_process()
    # Execute preprocess
    features_target_fin = (config['TRAIN']['CAT_FEATURES'] +
                           config['TRAIN']['BINARY_FEATURES'] +
                           config['TRAIN']['DISCRETE_FEATURES'] +
                           config['TRAIN']['CONT_FEATURES'] +
                           [config['VARIABLES']['TARGET']])
    if USE_TYPE == 'train':
        # Divide train and test
        X_train, X_test, X_val = split_train_val_test(df_features_churn)
        # Features encoder
        X_train, X_test, X_val = feature_encoder_train(X_train, X_test, X_val)
        X_train.to_csv(f"s3://{S3_BUCKET}/{save_path}/data_train/train.csv", index=False)
        X_test.to_csv(f"s3://{S3_BUCKET}/{save_path}/data_test/test.csv", index=False)
        X_val.to_csv(f"s3://{S3_BUCKET}/{save_path}/data_val/val.csv", index=False)
    else:
        X_pred = df_features_churn[features_target_fin[:-1]]
        X_pred = feature_encoder_predict(X_pred)
        X_pred[config['PREPROCESS']['VARIABLES_CCV']] = df_features_churn[config['PREPROCESS']['VARIABLES_CCV']]
        X_pred['cid'] = df_features_churn['cid']
        X_pred['ccv'] = (df_features_churn[config['PREPROCESS']['VARIABLES_CCV'][0]] +
                         df_features_churn[config['PREPROCESS']['VARIABLES_CCV'][1]] +
                         df_features_churn[config['PREPROCESS']['VARIABLES_CCV'][2]])
        X_pred.to_csv(f"s3://{S3_BUCKET}/{save_path}/data_for_prediction.csv", index=False)
