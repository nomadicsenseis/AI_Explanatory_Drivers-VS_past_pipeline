"""
Evaluation Step for BLVM.
"""
from subprocess import check_call
from sys import executable

STEP = "EVALUATION"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

if __name__ == "__main__":
    import argparse
    import logging
    from json import dumps as json_dumps
    from json import loads as json_loads
    from pathlib import Path

    import utils
    from boto3 import resource as s3_resource

    SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
    SAGEMAKER_LOGGER.setLevel(logging.INFO)
    SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())

    CONFIG = utils.read_config_data()
    CONFIG_STEP = CONFIG.get(f"{STEP}_STEP")
    CONFIG_GEN = CONFIG.get("GENERAL")

    class Arguments(utils.AbstractArguments):
        """Class to define the arguments used in the main functionality."""

        def __init__(self):
            """Class constructor."""
            super().__init__()
            parser = argparse.ArgumentParser(description=f"Inputs for {STEP} step.")
            parser.add_argument("--s3_bucket", type=str)
            parser.add_argument("--s3_path", type=str)
            parser.add_argument("--str_execution_date", type=str)
            parser.add_argument("--model_type", choices=["bl", "blv"], type=str)
            parser.add_argument("--is_last_date", type=str, default="1")
            parser.add_argument("--is_retrain_required", type=str, default="1")

            self.args = parser.parse_args()

    arguments = Arguments()
    args = arguments.get_arguments()
    path, _, _, _ = utils.get_path_to_read_and_date(
        read_last_date=bool(int(args.is_last_date)),
        bucket=args.s3_bucket,
        key=f"{args.s3_path}/300_train_step/metrics",
        partition_date=args.str_execution_date.replace("-", ""),
    )
    SAGEMAKER_LOGGER.info("userlog: Read date path %s.", path)
    previous_path, _, _, _ = utils.get_path_to_read_and_date(
        read_last_date=bool(int(args.is_last_date)),
        bucket=args.s3_bucket,
        key=f"{args.s3_path}/300_train_step/metrics",
        partition_date=args.str_execution_date.replace("-", ""),
        n_partition=2,
    )
    SAGEMAKER_LOGGER.info("userlog: Read previous date path %s.", previous_path)
    notfoundpreviousdate = "_notfoundpreviousdate" in previous_path

    s3 = s3_resource("s3")
    key_path = path.replace("s3://", "").replace(f"{args.s3_bucket}/", "")
    metrics_obj = s3.Object(args.s3_bucket, f"{key_path}/opt_metrics.json")
    metrics = json_loads(metrics_obj.get()["Body"].read().decode("utf-8"))
    report_dict = {
        "binary_classification_metrics": {
            "confusion_matrix": {
                "0": {
                    "0": str(metrics.get("tn_test")),
                    "1": str(metrics.get("fp_test")),
                },
                "1": {
                    "0": str(metrics.get("fn_test")),
                    "1": str(metrics.get("tp_test")),
                },
            },
            "recall": {
                "value": metrics.get("recall_test_l1"),
            },
            "precision": {
                "value": metrics.get("prec_test_l1"),
            },
            "accuracy": {"value": metrics.get("acc_test")},
            "true_positive_rate": {
                "value": str(
                    round(
                        int(metrics.get("tp_test"))
                        / (int(metrics.get("tp_test")) + int(metrics.get("fn_test"))),
                        5,
                    )
                ),
            },
            "true_negative_rate": {
                "value": str(
                    round(
                        int(metrics.get("tn_test"))
                        / (int(metrics.get("tn_test")) + int(metrics.get("fp_test"))),
                        5,
                    )
                ),
            },
            "false_positive_rate": {
                "value": str(
                    round(
                        int(metrics.get("fp_test"))
                        / (int(metrics.get("fp_test")) + int(metrics.get("tn_test"))),
                        5,
                    )
                ),
            },
            "false_negative_rate": {
                "value": str(
                    round(
                        int(metrics.get("fn_test"))
                        / (int(metrics.get("tp_test")) + int(metrics.get("fn_test"))),
                        5,
                    )
                ),
            },
            "auc": {
                "value": metrics.get("roc_test"),
            },
            "f1": {
                "value": metrics.get("f1_test_l1"),
            },
            "recall_l0": {
                "value": metrics.get("recall_test_l0"),
            },
            "precision_l0": {
                "value": metrics.get("prec_test_l0"),
            },
            "f1_l0": {
                "value": metrics.get("f1_test_l0"),
            },
        }
    }

    output_dir = "/opt/ml/processing/train_step_evaluation_report"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/train_step_evaluation_report.json"
    with open(evaluation_path, "w") as f:
        f.write(json_dumps(report_dict))

    if bool(int(args.is_retrain_required)):
        SAGEMAKER_LOGGER.info("userlog: Retrain is required (mandatory).")
    else:
        if notfoundpreviousdate:
            SAGEMAKER_LOGGER.info(
                "userlog: Not found previous execution to check with the actual one."
            )
        else:
            SAGEMAKER_LOGGER.info(
                "userlog: Retrain is not required (not mandatory). Checking if new one is better than the previous one..."
            )
            actual_f1_mean = (metrics.get("f1_test_l1") + metrics.get("f1_test_l0")) / 2
            previous_key_path = previous_path.replace("s3://", "").replace(
                f"{args.s3_bucket}/", ""
            )
            previous_metrics_obj = s3.Object(
                args.s3_bucket, f"{previous_key_path}/opt_metrics.json"
            )
            previous_metrics = json_loads(
                previous_metrics_obj.get()["Body"].read().decode("utf-8")
            )
            previous_f1_mean = (
                previous_metrics.get("f1_test_l1") + previous_metrics.get("f1_test_l0")
            ) / 2
            growth_rate = 100 * (actual_f1_mean - previous_f1_mean) / previous_f1_mean
            if growth_rate >= float(CONFIG_STEP.get("MIN_GROWTH_RATE")):
                SAGEMAKER_LOGGER.info(
                    f"userlog: New model is better than previous one. Growth rate is {growth_rate}"
                )
            else:
                SAGEMAKER_LOGGER.info(
                    "userlog: New model is worse than previous one. Moving files to a non_approved_models path..."
                )
                non_approved_models = [
                    f"200_preprocess_step/train/models/{args.str_execution_date.replace('-', '')}/categorical_encoder.pkl",
                    f"200_preprocess_step/train/models/{args.str_execution_date.replace('-', '')}/null_imputer.pkl",
                    f"200_preprocess_step/train/models/{args.str_execution_date.replace('-', '')}/iata_pct.csv",
                    f"300_train_step/model/{args.str_execution_date.replace('-', '')}/clf.pkl",
                ]
                for nam in non_approved_models:
                    s3.Bucket(args.s3_bucket).Object(
                        f"{args.s3_path}/non_approved_models/{nam}"
                    ).copy_from(f"{args.s3_path}/{nam}")
                    s3.Bucket(args.s3_bucket).Object(f"{args.s3_path}/{nam}").delete()
