import json
import pickle
import joblib
from subprocess import check_call
from sys import executable

STEP = "PREPROCESS"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

import category_encoders as ce
import sklearn.preprocessing as prep
import warnings
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

import argparse
import logging
import pandas as pd
import pickle
import time
from pandas import DataFrame

import boto3
import utils
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())


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


def cabin_preprocesser(x):
    return 0 if pd.isna(x) else 1


def feature_processer(df: DataFrame, use_type='predict', y_train=None) -> DataFrame:
    """
    This function processes the features of the input DataFrame depending on the mode (prediction or training).
    If in prediction mode, it applies a pre-saved pipeline transformation.
    If in training mode, it fits and applies the transformation pipeline, and saves the pipeline for future use.

    Parameters:
    df (DataFrame): The input DataFrame to be processed.
    use_type (str): The mode, either 'predict' or 'train'. Default is 'predict'.
    y_train (Series): The target variable Series if in 'train' mode.

    Returns:
    DataFrame: The processed DataFrame.
    """

    # In prediction mode
    if use_type == 'predict':
        # Get the path to the saved transformation pipeline
        model_path, _, _, _ = utils.get_path_to_read_and_date(
            read_last_date=bool(int(IS_LAST_DATE)),
            bucket=S3_BUCKET,
            key=path_read,
            partition_date=STR_EXECUTION_DATE,
        )
        # Remove s3 info path
        if 's3://' in model_path:
            model_path = model_path.split('//')[1].replace(f"{S3_BUCKET}/", '')
        SAGEMAKER_LOGGER.info(f"userlog: path for models: {model_path + '/models/'}")

        # Read the transformation pipeline
        pipeline = (
            s3_resource.Object(S3_BUCKET, f"{model_path}/models/{config['PREPROCESS']['PIPELINE_NAME']}")
            .get()
        )

        # Load the pipeline and apply transformation
        pipe = pickle.loads(pipeline["Body"].read())
        time.sleep(30)
        X = pipe.transform(df)
        X = pd.DataFrame(X, columns=get_names_from_pipeline(pipe.named_steps['preprocessor']))

    # In training mode
    else:
        y_train = pd.DataFrame(y_train, columns=[config.get("VARIABLES").get('TARGET')])
        SAGEMAKER_LOGGER.info(f'Processing X_train {X_train.shape} ')
        SAGEMAKER_LOGGER.info(f'Processing y_train {y_train.shape} ')

        # Transform 'Cabin' into a binary variable
        cabin_transformer = prep.FunctionTransformer(cabin_preprocesser)
        cabin_pipeline = Pipeline(steps=[('cabin_preprocesser', cabin_transformer)])

        # Define transformations for the columns
        numeric_features = ['Age', 'Fare']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = ['Embarked', 'Sex', 'Pclass', 'Cabin', 'Title']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Combine all transformations using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('cabin', cabin_pipeline, ['Cabin'])])

        # Fit and apply the transformation pipeline, then save it
        pipe = Pipeline(steps=[('preprocessor', preprocessor)])
        X = pipe.fit_transform(df, y_train)
        s3_resource.Object(S3_BUCKET, f"{save_path}/models/{config['PREPROCESS']['PIPELINE_NAME']}").put(
            Body=pickle.dumps(pipe)
        )
        X = pd.DataFrame(X, columns=get_names_from_pipeline(preprocessor))
        SAGEMAKER_LOGGER.info(f'Columns names out prep {get_names_from_pipeline(preprocessor)}')

    X.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    X[config['VARIABLES_ETL']['ID']] = df[config['VARIABLES_ETL']['ID']].copy()

    return X


def get_names_from_pipeline(preprocessor):
    """
    This function returns the names of the columns that are outputted by the preprocessor.

    Parameters:
    preprocessor (ColumnTransformer): The preprocessor to get output column names from.

    Returns:
    output_columns (list): List of the output column names.
    """
    output_columns = []

    # For each transformer in the preprocessor
    for name, transformer, cols in preprocessor.transformers_:
        # If the transformer is 'drop' or columns are 'drop', continue to the next transformer
        if transformer == 'drop' or cols == 'drop':
            continue

        # If the transformer is a Pipeline, get the last step of the pipeline
        if isinstance(transformer, Pipeline):
            transformer = transformer.steps[-1][1]  # get the last step of the pipeline

        # Depending on the type of the transformer, get the transformed column names
        if isinstance(transformer, ce.TargetEncoder):
            names = [f'{col}_target_enc' for col in cols]
            output_columns += names
        elif isinstance(transformer, ce.WOEEncoder):
            names = [f'{col}_woe_enc' for col in cols]
            output_columns += names
        elif isinstance(transformer, prep.OneHotEncoder):
            names = [f'{col}_enc' for col in transformer.get_feature_names_out(cols)]
            output_columns += names
        else:
            output_columns += cols

    # Return the list of output column names
    return output_columns


def split_train_val_test(X: DataFrame, target) -> object:
    # Split between train, validation and test
    X_traintest, X_val, y_traintest, y_val = train_test_split(X, target, stratify=target, test_size=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X_traintest, y_traintest, stratify=y_traintest, test_size=0.2)
    return X_train, X_test, X_val, y_train, y_test, y_val


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
    src_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/00_etl_step/{USE_TYPE}/{year}{month}{day}/titanic.csv"
    out_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/01_preprocess_step/{USE_TYPE}/{year}{month}{day}"

    # Read data
    df_features = pd.read_csv(src_path)

    # Execute preprocess
    in_features_train = config.get("VARIABLES_ETL").get('COLUMNS_TO_SAVE') + \
                        [config.get("VARIABLES").get('ID')]

    if USE_TYPE == 'train':

        # Divide train and test
        X_train, X_test, X_val, y_train, y_test, y_val = split_train_val_test(df_features[in_features_train],
                                                                              df_features[config.get("VARIABLES").get(
                                                                                  'TARGET')])
        # Features encoder
        SAGEMAKER_LOGGER.info(f"X_train, X_test, X_val, y_train, y_test, y_val: {X_train.shape}, {X_test.shape}, "
                              f"{X_val.shape}, {y_train.shape}, {y_test.shape}, {y_val.shape}")

        X_train = feature_processer(X_train, use_type='train', y_train=y_train)

        X_test = feature_processer(X_test)
        SAGEMAKER_LOGGER.info(f'Processing X_test, shape {X_test.shape}')
        X_val = feature_processer(X_val)
        SAGEMAKER_LOGGER.info(f'Processing X_val, shape {X_val.shape}')

        X_train.to_csv(f"{out_path}/data_train/X_train.csv", index=False)
        X_test.to_csv(f"{out_path}/data_test/X_test.csv", index=False)
        X_val.to_csv(f"{out_path}/data_val/X_val.csv", index=False)

        y_train.to_csv(f"{out_path}/data_train/y_train.csv", index=False)
        y_test.to_csv(f"{out_path}/data_test/y_test.csv", index=False)
        y_val.to_csv(f"{out_path}/data_val/y_val.csv", index=False)

    else:
        X_pred = df_features[in_features_predict]
        SAGEMAKER_LOGGER.info(f"userlog: feature_processer_predict pre: {str(X_pred.shape)}")
        X_pred = feature_processer(X_pred)
        SAGEMAKER_LOGGER.info(f"userlog: feature_processer_predict post: {str(X_pred.shape)}")
        X_pred.to_csv(f"{out_path}/data_for_prediction.csv", index=False)
