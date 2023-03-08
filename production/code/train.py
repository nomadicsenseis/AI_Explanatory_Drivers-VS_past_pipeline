"""
Train Step for BLVM.
"""
from subprocess import check_call
from sys import executable

STEP = "TRAIN"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

import argparse
import logging
from json import dumps as jdumps
from os import environ
from os.path import join as path_join
from pickle import dumps as pkl_dumps

import plots
import train_utils as tutils
import utils
from boto3 import resource
from joblib import dump as jl_dump
from optuna import create_study
from pandas import read_csv
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
        parser.add_argument(
            "--model_dir", type=str, default=environ["SM_MODEL_DIR"]
        )

        self.args = parser.parse_args()

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
    STR_EXECUTION_DATE = args.str_execution_date
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]
    # Paths
    read_path = f"{S3_PATH_WRITE}/01_preprocess_step/train/{year}{month}{day}"
    save_path = f"{S3_PATH_WRITE}/02_train_step/{year}{month}{day}"
    SAGEMAKER_LOGGER.info("userlog: Read date path %s.", read_path)
    # Read data
    X_train = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_train/train.csv")
    X_test = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_test/test.csv")
    X_val = read_csv(f"s3://{S3_BUCKET}/{read_path}/data_val/val.csv")
    # Variables
    features = (config['TRAIN']['CAT_FEATURES'] +
               config['TRAIN']['BINARY_FEATURES'] +
               config['TRAIN']['DISCRETE_FEATURES'] +
               config['TRAIN']['CONT_FEATURES'] +
               config['TRAIN']['EXTRA_FEATURES'])
    for feat in config['TRAIN']['ELIMINATE_FEATURES']:
        features.remove(feat)
    # Estimator
    estimator, parameters = tutils.get_model_and_params(
        model_name=config['TRAIN']['MODEL_NAME']
    )
    model_optimizer = tutils.HPTClassifier(
        clf=estimator,
        params=parameters,
        train_set=X_train,
        validation_set=X_val,
        feature_cols=features,
        target_col=config['VARIABLES']['TARGET'],
        opt_metric=config['TRAIN']['OPT_METRICS'],
    )
    hpt_study = create_study(direction="maximize")
    hpt_study.optimize(model_optimizer, n_trials=config['TRAIN']['N_TRIALS'], n_jobs=16)
    optimized_model = estimator(**hpt_study.best_params)
    optimized_model.fit(X_train[features], X_train[config['VARIABLES']['TARGET']])
    # Test
    acc1 = accuracy_score(
        y_true=X_train[config['VARIABLES']['TARGET']],
        y_pred=optimized_model.predict(X_train[features]),
    )
    acc2 = accuracy_score(
        y_true=X_test[config['VARIABLES']['TARGET']],
        y_pred=optimized_model.predict(X_test[features]),
    )
    SAGEMAKER_LOGGER.info(f"userlog: Accuracy train IS... : {str(acc1)}")
    SAGEMAKER_LOGGER.info(f"userlog: Accuracy train IS... : {str(acc2)}")
    # Create global dataframe for final train
    X = X_train.append(X_test.append(X_val))
    # Retrain with all data
    model = estimator(**hpt_study.best_params)
    model.fit(X[features], X[config['VARIABLES']['TARGET']])
    SAGEMAKER_LOGGER.info(f"userlog: Feature columns are: {features}")
    SAGEMAKER_LOGGER.info(f"userlog: Generating plots and metrics...")
    cfm_train, cfm_validation, cfm_test = plots.generate_confusion_matrix(
        model=optimized_model,
        train_set=X_train,
        val_set=X_val,
        test_set=X_test,
        target_col=config['VARIABLES']['TARGET'],
        feat_col=features,
        s3_bucket_name=S3_BUCKET,
        s3_path=f"{save_path}/metrics",
    )
    plots.generate_metric_plots(
        model=optimized_model,
        train_set=X_train,
        val_set=X_val,
        test_set=X_test,
        target_col=config['VARIABLES']['TARGET'],
        feat_col=features,
        s3_bucket_name=S3_BUCKET,
        s3_path=f"{save_path}/metrics",
    )

    s3_resource = resource("s3")
    fitted_clf_model = pkl_dumps(model)
    # jl_dump(model, path_join(args.model_dir, "model.joblib"))
    s3_resource.Object(
        S3_BUCKET,
        f"{save_path}/model/{config['TRAIN']['MODEL_NAME_SAVE']}",
    ).put(Body=fitted_clf_model)

    opt_metrics = {}
    clf_metrics = {}
    for idx, cfm_name in enumerate(["TN", "FP", "FN", "TP"]):
        opt_metrics[f"{cfm_name.lower()}_train"] = str(cfm_train[idx])
        opt_metrics[f"{cfm_name.lower()}_validation"] = str(cfm_validation[idx])
        opt_metrics[f"{cfm_name.lower()}_test"] = str(cfm_test[idx])

    dfs = [("train", X_train), ("validation", X_val), ("test", X_test)]
    for name, dataset in dfs:
        opt_metrics[f"acc_{name}"] = accuracy_score(
            y_true=dataset[config['VARIABLES']['TARGET']],
            y_pred=optimized_model.predict(dataset[features]),
        )
        opt_metrics[f"roc_{name}"] = roc_auc_score(
            y_true=dataset[config['VARIABLES']['TARGET']],
            y_score=optimized_model.predict_proba(dataset[features])[:, 1],
        )
        opt_metrics[f"prec_{name}"] = precision_score(
            y_true=dataset[config['VARIABLES']['TARGET']],
            y_pred=optimized_model.predict(dataset[features]),
            average="binary",
            pos_label=1,
        )
        opt_metrics[f"recall_{name}"] = recall_score(
            y_true=dataset[config['VARIABLES']['TARGET']],
            y_pred=optimized_model.predict(dataset[features]),
            average="binary",
            pos_label=1,
        )
        opt_metrics[f"f1_{name}"] = f1_score(
            y_true=dataset[config['VARIABLES']['TARGET']],
            y_pred=optimized_model.predict(dataset[features]),
            average="binary",
            pos_label=1,
        )

    clf_metrics["acc_clf"] = accuracy_score(
        y_true=X[config['VARIABLES']['TARGET']],
        y_pred=model.predict(X[features]),
    )
    clf_metrics["roc_clf"] = roc_auc_score(
        y_true=X[config['VARIABLES']['TARGET']],
        y_score=model.predict_proba(X[features])[:, 1],
    )
    clf_metrics["prec_clf"] = precision_score(
        y_true=X[config['VARIABLES']['TARGET']],
        y_pred=model.predict(X[features]),
        average="binary",
        pos_label=1,
    )
    clf_metrics["recall_clf"] = recall_score(
        y_true=X[config['VARIABLES']['TARGET']],
        y_pred=model.predict(X[features]),
        average="binary",
        pos_label=1,
    )
    clf_metrics["f1_clf"] = f1_score(
        y_true=X[config['VARIABLES']['TARGET']],
        y_pred=model.predict(X[features]),
        average="binary",
        pos_label=1,
    )

    clf_metrics_json = jdumps(clf_metrics)
    s3_resource.Object(
        S3_BUCKET,
        f"{save_path}/metrics/clf_metrics.json",
    ).put(Body=(bytes(clf_metrics_json.encode("UTF-8"))))
    opt_metrics_json = jdumps(opt_metrics)
    s3_resource.Object(
        args.s3_bucket,
        f"{save_path}/metrics/opt_metrics.json",
    ).put(Body=(bytes(opt_metrics_json.encode("UTF-8"))))