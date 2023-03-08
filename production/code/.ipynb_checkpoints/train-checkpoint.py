"""
Train Step for BLVM.
"""
from subprocess import check_call
from sys import executable

STEP = "TRAIN"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

# TODO: Model has to be saved also in model-dir variable to use it in Model Creation
# TODO: check: https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#prepare-a-scikit-learn-training-script

if __name__ == "__main__":
    import argparse
    import logging
    from datetime import date
    from json import dumps as jdumps
    from os import environ
    from os.path import join as path_join
    from pickle import dumps as pkl_dumps

    import plots
    import train_utils as tutils
    import utils
    from boto3 import resource
    from dateutil.relativedelta import relativedelta
    from joblib import dump as jl_dump
    from numpy import where
    from optuna import create_study
    from pandas import read_csv
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
    SAGEMAKER_LOGGER.setLevel(logging.INFO)
    SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())

    CONFIG = utils.read_config_data()
    CONFIG_STEP = CONFIG.get(f"{STEP}_STEP")
    CONFIG_GEN = CONFIG.get("GENERAL")

    OUTPUT_OBJECT_FEAT = [
        CONFIG_GEN.get("SCALES_NAME"),
        CONFIG_GEN.get("TIME_FLIGHT_NAME"),
        CONFIG_GEN.get("CLOSE_HOLIDAYS_NAME"),
        CONFIG_GEN.get("CAT_COMPANION_NUMBER_NAME"),
        CONFIG_GEN.get("BOOL_COMPANION_NUMBER_NAME"),
        CONFIG_GEN.get("BOOL_DESTINATION_TIMES_NAME"),
        CONFIG_GEN.get("BOOL_DESTINATION_TIMES_LQ_NAME"),
        CONFIG_GEN.get("FIRST_TRAVEL_NAME"),
        CONFIG_GEN.get("BAGS_NUMBER_NAME"),
        CONFIG_GEN.get("NON_WORKING_MAILS_NAME"),
        CONFIG_GEN.get("HAUL_NAME"),
        CONFIG_GEN.get("CLASS_NAME"),
        CONFIG_GEN.get("RESIDENT_NAME"),
        CONFIG_GEN.get("TIER_NAME"),
        CONFIG_GEN.get("SALES_CHANNEL_NAME"),
        CONFIG_GEN.get("WEEKDAY_NAME"),
        CONFIG_GEN.get("AVIOS_NAME"),
        CONFIG_GEN.get("SEATS_NUMBER_NAME"),
        CONFIG_GEN.get("FLEX_FF_NAME"),
        CONFIG_GEN.get("FARE_FAMILY_NAME"),
        CONFIG_GEN.get("IS_CORPORATE_NAME"),
    ]

    OUTPUT_OBJECT_NUM = [
        CONFIG_GEN.get("PCT_REP3M_VS_REP_NAME"),
        CONFIG_GEN.get("BAGS_PAYMENT_NAME"),
        CONFIG_GEN.get("PCT_REP_VS_TRAVELS_NAME"),
        CONFIG_GEN.get("PAYMENT_NAME"),
        CONFIG_GEN.get("DAYS_IN_DESTINATION_NAME"),
        CONFIG_GEN.get("SEATS_PAYMENT_NAME"),
        CONFIG_GEN.get("DESTINATION_TIMES_NAME"),
        CONFIG_GEN.get("TRAVEL_TIMES_NAME"),
        CONFIG_GEN.get("DESTINATION_TIMES_LQ_NAME"),
        CONFIG_GEN.get("TRAVEL_TIMES_LQ_NAME"),
        CONFIG_GEN.get("COMPANION_NUMBER_NAME"),
        CONFIG_GEN.get("WEEKNUM_NAME"),
        CONFIG_GEN.get("MONTH_NAME"),
    ]

    OUTPUT_IATA_PCT_BLV = [
        CONFIG_GEN.get("PCT_IATA_BUSINESS_NAME"),
        CONFIG_GEN.get("PCT_IATA_LEISURE_NAME"),
        CONFIG_GEN.get("PCT_IATA_VFR_NAME"),
    ]

    OUTPUT_IATA_PCT_BL = [
        CONFIG_GEN.get("PCT_IATA_BUSINESS_NAME"),
        CONFIG_GEN.get("PCT_IATA_LEISURE_NAME"),
    ]

    # TODO: correlated_columns are better to not be calculated.
    CORRELATED_COLUMNS = [
        CONFIG_GEN.get("PCT_IATA_LEISURE_NAME"),
        CONFIG_GEN.get("CAT_COMPANION_NUMBER_NAME"),
        CONFIG_GEN.get("BAGS_PAYMENT_NAME"),
        CONFIG_GEN.get("WEEKNUM_NAME"),
        CONFIG_GEN.get("SEATS_PAYMENT_NAME"),
        CONFIG_GEN.get("PCT_REP3M_VS_REP_NAME"),
        CONFIG_GEN.get("FIRST_TRAVEL_NAME"),
        CONFIG_GEN.get("DESTINATION_TIMES_LQ_NAME"),
        CONFIG_GEN.get("TRAVEL_TIMES_NAME"),
        CONFIG_GEN.get("BOOL_COMPANION_NUMBER_NAME"),
    ]

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
            parser.add_argument("--month_threshold_split", type=str, default="4")
            parser.add_argument("--trials", type=str)
            parser.add_argument(
                "--model_dir", type=str, default=environ["SM_MODEL_DIR"]
            )

            self.args = parser.parse_args()

    def main():
        """Main functionality of the script."""

        # DEFINE ARGUMENTS
        SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
        arguments = Arguments()
        arguments.info(logger=SAGEMAKER_LOGGER)
        args = arguments.get_arguments()
        target_column = (
            CONFIG_GEN.get("BLV_TARGET_COL")
            if args.model_type == "blv"
            else CONFIG_GEN.get("BL_TARGET_COL")
        )
        # READ DATA
        path, year, month, day = utils.get_path_to_read_and_date(
            read_last_date=bool(int(args.is_last_date)),
            bucket=args.s3_bucket,
            key=f"{args.s3_path}/200_preprocess_step/train/data",
            partition_date=args.str_execution_date.replace("-", ""),
        )
        SAGEMAKER_LOGGER.info("userlog: Read date path %s.", path)
        dataframe = read_csv(f"{path}/preprocess_data.csv")

        threshold_split = int(
            str(
                date(int(year), int(month), int(day))
                - relativedelta(months=int(args.month_threshold_split))
            ).replace("-", "")
        )
        dataframe["date_creation_pnr_resiber"] = where(
            dataframe["date_creation_pnr_resiber"].isnull(),
            dataframe[CONFIG_GEN.get("DEPARTURE_TIME_NAME")].str.replace("-", ""),
            dataframe["date_creation_pnr_resiber"],
        )
        dataframe["date_creation_pnr_resiber"] = (
            dataframe["date_creation_pnr_resiber"].str.split(" ").str[0].astype(int)
        )
        data = dataframe[dataframe["date_creation_pnr_resiber"] >= threshold_split]
        train_df = dataframe[dataframe["date_creation_pnr_resiber"] < threshold_split]
        validation_df, test_df = train_test_split(
            data, test_size=0.7, stratify=data[target_column]
        )
        feature_columns = [
            *OUTPUT_OBJECT_FEAT,
            *OUTPUT_OBJECT_NUM,
        ] + (OUTPUT_IATA_PCT_BLV if args.model_type == "blv" else OUTPUT_IATA_PCT_BL)
        feature_columns = list(set(feature_columns) - set(CORRELATED_COLUMNS))
        estimator, parameters = tutils.get_model_and_params(
            model_name=CONFIG_STEP.get("CLASSIFIER_NAME")
        )
        model_optimizer = tutils.HPTClassifier(
            clf=estimator,
            params=parameters,
            train_set=train_df,
            validation_set=validation_df,
            feature_cols=feature_columns,
            target_col=target_column,
            opt_metric="f1-score",
        )
        hpt_study = create_study(direction="maximize")
        hpt_study.optimize(model_optimizer, n_trials=int(args.trials))

        optimized_model = estimator(**hpt_study.best_params)
        optimized_model.fit(train_df[feature_columns], train_df[target_column])

        model = estimator(**hpt_study.best_params)
        model.fit(dataframe[feature_columns], dataframe[target_column])
        SAGEMAKER_LOGGER.info("userlog: Feature columns are", feature_columns)

        cfm_train, cfm_validation, cfm_test = plots.generate_confusion_matrix(
            model=optimized_model,
            train_set=train_df,
            val_set=validation_df,
            test_set=test_df,
            target_col=target_column,
            feat_col=feature_columns,
            s3_bucket_name=args.s3_bucket,
            s3_path=f"{args.s3_path}/300_train_step/metrics/{year}{month}{day}",
        )
        plots.generate_metric_plots(
            model=optimized_model,
            train_set=train_df,
            val_set=validation_df,
            test_set=test_df,
            target_col=target_column,
            feat_col=feature_columns,
            s3_bucket_name=args.s3_bucket,
            s3_path=f"{args.s3_path}/300_train_step/metrics/{year}{month}{day}",
        )

        s3_resource = resource("s3")
        fitted_clf_model = pkl_dumps(model)
        jl_dump(model, path_join(args.model_dir, "model.joblib"))
        s3_resource.Object(
            args.s3_bucket,
            f"{args.s3_path}/300_train_step/model/{year}{month}{day}/{CONFIG_GEN.get('SAVING_MODEL_NAME')}.pkl",
        ).put(Body=fitted_clf_model)

        opt_metrics = {}
        clf_metrics = {}
        # TODO: Make the cfm list more general adding into a config
        for idx, cfm_name in enumerate(["TN", "FP", "FN", "TP"]):
            opt_metrics[f"{cfm_name.lower()}_train"] = str(cfm_train[idx])
            opt_metrics[f"{cfm_name.lower()}_validation"] = str(cfm_validation[idx])
            opt_metrics[f"{cfm_name.lower()}_test"] = str(cfm_test[idx])

        dfs = [("train", train_df), ("validation", validation_df), ("test", test_df)]
        for name, dataset in dfs:
            opt_metrics[f"acc_{name}"] = accuracy_score(
                y_true=dataset[target_column],
                y_pred=optimized_model.predict(dataset[feature_columns]),
            )
            opt_metrics[f"roc_{name}"] = roc_auc_score(
                y_true=dataset[target_column],
                y_score=optimized_model.predict_proba(dataset[feature_columns])[:, 1],
            )
            opt_metrics[f"prec_{name}_l0"] = precision_score(
                y_true=dataset[target_column],
                y_pred=optimized_model.predict(dataset[feature_columns]),
                average="binary",
                pos_label=0,
            )
            opt_metrics[f"prec_{name}_l1"] = precision_score(
                y_true=dataset[target_column],
                y_pred=optimized_model.predict(dataset[feature_columns]),
                average="binary",
                pos_label=1,
            )
            opt_metrics[f"recall_{name}_l0"] = recall_score(
                y_true=dataset[target_column],
                y_pred=optimized_model.predict(dataset[feature_columns]),
                average="binary",
                pos_label=0,
            )
            opt_metrics[f"recall_{name}_l1"] = recall_score(
                y_true=dataset[target_column],
                y_pred=optimized_model.predict(dataset[feature_columns]),
                average="binary",
                pos_label=1,
            )
            opt_metrics[f"f1_{name}_l0"] = f1_score(
                y_true=dataset[target_column],
                y_pred=optimized_model.predict(dataset[feature_columns]),
                average="binary",
                pos_label=0,
            )
            opt_metrics[f"f1_{name}_l1"] = f1_score(
                y_true=dataset[target_column],
                y_pred=optimized_model.predict(dataset[feature_columns]),
                average="binary",
                pos_label=1,
            )

        clf_metrics["acc_clf"] = accuracy_score(
            y_true=dataframe[target_column],
            y_pred=model.predict(dataframe[feature_columns]),
        )
        clf_metrics["roc_clf"] = roc_auc_score(
            y_true=dataframe[target_column],
            y_score=model.predict_proba(dataframe[feature_columns])[:, 1],
        )
        clf_metrics["prec_clf_l0"] = precision_score(
            y_true=dataframe[target_column],
            y_pred=model.predict(dataframe[feature_columns]),
            average="binary",
            pos_label=0,
        )
        clf_metrics["prec_clf_l1"] = precision_score(
            y_true=dataframe[target_column],
            y_pred=model.predict(dataframe[feature_columns]),
            average="binary",
            pos_label=1,
        )
        clf_metrics["recall_clf_l0"] = recall_score(
            y_true=dataframe[target_column],
            y_pred=model.predict(dataframe[feature_columns]),
            average="binary",
            pos_label=0,
        )
        clf_metrics["recall_clf_l1"] = recall_score(
            y_true=dataframe[target_column],
            y_pred=model.predict(dataframe[feature_columns]),
            average="binary",
            pos_label=1,
        )
        clf_metrics["f1_clf_l0"] = f1_score(
            y_true=dataframe[target_column],
            y_pred=model.predict(dataframe[feature_columns]),
            average="binary",
            pos_label=0,
        )
        clf_metrics["f1_clf_l1"] = f1_score(
            y_true=dataframe[target_column],
            y_pred=model.predict(dataframe[feature_columns]),
            average="binary",
            pos_label=1,
        )

        clf_metrics_json = jdumps(clf_metrics)
        s3_resource.Object(
            args.s3_bucket,
            f"{args.s3_path}/300_train_step/metrics/{year}{month}{day}/clf_metrics.json",
        ).put(Body=(bytes(clf_metrics_json.encode("UTF-8"))))
        opt_metrics_json = jdumps(opt_metrics)
        s3_resource.Object(
            args.s3_bucket,
            f"{args.s3_path}/300_train_step/metrics/{year}{month}{day}/opt_metrics.json",
        ).put(Body=(bytes(opt_metrics_json.encode("UTF-8"))))

    main()
