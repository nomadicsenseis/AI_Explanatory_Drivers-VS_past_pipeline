"""
Explainer Step for BLVM.
"""
from subprocess import check_call
from sys import executable

STEP = "EXPLAINER"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

import argparse
import logging
from io import BytesIO
from pickle import loads as pkl_loads

import utils
from boto3 import resource
from matplotlib import pyplot as plt
from pandas import read_csv
from shap import TreeExplainer, summary_plot


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

    SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
    SAGEMAKER_LOGGER.setLevel(logging.INFO)
    SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())

    config = utils.read_config_data()

    def main() -> None:
        """Main functionality of the script."""

        # DEFINE ARGUMENTS
        SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
        arguments = Arguments()
        arguments.info(logger=SAGEMAKER_LOGGER)
        args = arguments.get_arguments()
        S3_BUCKET = args.s3_bucket
        S3_PATH_WRITE = args.s3_path_write
        STR_EXECUTION_DATE = args.str_execution_date
        date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
        year, month, day = date[:4], date[4:6], date[6:]
        read_path = f"{S3_PATH_WRITE}/01_preprocess_step/train/{year}{month}{day}"
        save_path = f"{S3_PATH_WRITE}/02_train_step/{year}{month}{day}"
        # READ DATA
        SAGEMAKER_LOGGER.info("userlog: Read date path %s.", read_path)
        df = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_test/test.csv")
        s3_resource = resource("s3")
        fitted_clf_model = (
            s3_resource.Bucket(args.s3_bucket)
            .Object(f"{save_path}/model/{config['TRAIN']['MODEL_NAME_SAVE']}")
            .get()
        )
        clf_model = pkl_loads(fitted_clf_model["Body"].read())
        SAGEMAKER_LOGGER.info(f"Model read.")
        # Variables
        features = (config['TRAIN']['CAT_FEATURES'] +
                    config['TRAIN']['BINARY_FEATURES'] +
                    config['TRAIN']['DISCRETE_FEATURES'] +
                    config['TRAIN']['CONT_FEATURES'] +
                    config['TRAIN']['EXTRA_FEATURES'])
        for feat in config['TRAIN']['ELIMINATE_FEATURES']:
            features.remove(feat)
        # SHAP Values
        shapley_values = TreeExplainer(
            clf_model, output_names=["No Churner", "Churner"]
        ).shap_values(df[features].sample(3000)) # Sampling
        SAGEMAKER_LOGGER.info(f"Shap values computed.")
        plt.figure(figsize=(14, 8))
        summary_plot(
            shapley_values,
            df[features],
            feature_names=features,
            plot_type="bar",
            plot_size=(14, 8),
            class_names=["No Churner", "Churner"],
            show=False,
        )
        bar_type_shapley = BytesIO()
        plt.savefig(bar_type_shapley, format="png")
        bar_type_shapley.seek(0)
        s3_resource.Bucket(S3_BUCKET).put_object(
            Body=bar_type_shapley,
            ContentType="image/png",
            Key=f"{save_path}/metrics/bar_type_shapley.png",
        )

        plt.figure(figsize=(14, 8))
        summary_plot(
            shapley_values,
            df[features],
            feature_names=features,
            plot_type="dot",
            show=False,
        )
        dot_type_shapley = BytesIO()
        plt.savefig(dot_type_shapley, format="png")
        dot_type_shapley.seek(0)
        s3_resource.Bucket(S3_BUCKET).put_object(
            Body=dot_type_shapley,
            ContentType="image/png",
            Key=f"{save_path}/metrics/dot_type_shapley.png",
        )

    main()
