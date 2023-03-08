"""
Predict Step for BLVM.
"""
from subprocess import check_call
from sys import executable

STEP = "PREDICT"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

if __name__ == "__main__":
    import argparse
    import logging
    from pickle import loads as pkl_loads

    import utils
    from boto3 import resource
    from pandas import read_csv

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
            parser.add_argument("--machines", type=str, default="4")

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
        predicted_column = CONFIG_GEN.get("PREDICT_COLUMN_NAME")
        feature_columns = [
            *OUTPUT_OBJECT_FEAT,
            *OUTPUT_OBJECT_NUM,
        ] + (OUTPUT_IATA_PCT_BLV if args.model_type == "blv" else OUTPUT_IATA_PCT_BL)
        feature_columns = list(set(feature_columns) - set(CORRELATED_COLUMNS))
        SAGEMAKER_LOGGER.info(
            f"userlog: Feature columns are {'.'.join(feature_columns)}"
        )
        save_columns = [
            *CONFIG_GEN.get("KEY_COLUMNS"),
            *feature_columns,
            target_column,
            predicted_column,
            f"{predicted_column}_proba_label0",
            f"{predicted_column}_proba_label1",
        ]

        # READ MODEL
        model_name = CONFIG_GEN.get("SAVING_MODEL_NAME")
        model_path, _, _, _ = utils.get_path_to_read_and_date(
            read_last_date=bool(int(args.is_last_date)),
            bucket=args.s3_bucket,
            key=f"{args.s3_path}/300_train_step/model",
            partition_date=args.str_execution_date.replace("-", ""),
        )
        SAGEMAKER_LOGGER.info("userlog: Read model date path %s.", model_path)
        s3_resource = resource("s3")
        model_path = model_path.replace("s3://", "").replace(f"{args.s3_bucket}/", "")
        fitted_clf_model = (
            s3_resource.Bucket(args.s3_bucket)
            .Object(f"{model_path}/{model_name}.pkl")
            .get()
        )
        clf_model = pkl_loads(fitted_clf_model["Body"].read())

        # READ DATA | PREDICT
        path, year, month, day = utils.get_path_to_read_and_date(
            read_last_date=bool(int(args.is_last_date)),
            bucket=args.s3_bucket,
            key=f"{args.s3_path}/200_preprocess_step/predict/data",
            partition_date=args.str_execution_date.replace("-", ""),
        )
        for mach in range(1, int(args.machines) + 1):
            dataframe = read_csv(f"{path}/preprocess_data_{args.machines}_{mach}.csv")
            SAGEMAKER_LOGGER.info(
                "userlog: Read date path %s.",
                f"{path}/preprocess_data_{args.machines}_{mach}.csv with {dataframe.shape[0]} rows",
            )
            dataframe[predicted_column] = clf_model.predict(
                dataframe[clf_model.feature_names_in_]
            )
            probabilities = clf_model.predict_proba(
                dataframe[clf_model.feature_names_in_]
            )
            dataframe[f"{predicted_column}_proba_label0"] = probabilities[:, 0]
            dataframe[f"{predicted_column}_proba_label1"] = probabilities[:, 1]
            dataframe = dataframe[save_columns]
            dataframe[CONFIG_GEN.get("KEY_COLUMNS")[-1]] = (
                dataframe[CONFIG_GEN.get("KEY_COLUMNS")[-1]].astype(str).str.zfill(3)
            )  # TODO: Reformat code to read as str at the begining
            # SAVE DATA
            save_path = f"s3://{args.s3_bucket}/{args.s3_path}/310_predict_step/v1/{year}{month}{day}/predictions_{mach}.csv"
            SAGEMAKER_LOGGER.info(
                "userlog: Saving information for predict step in %s.", save_path
            )
            dataframe.to_csv(save_path, index=False)

    main()
