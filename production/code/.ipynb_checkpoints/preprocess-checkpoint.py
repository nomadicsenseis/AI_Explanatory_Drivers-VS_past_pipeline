"""
PREPROCESS Step for BLVM.
"""
from subprocess import check_call
from sys import executable
from typing import Any, Dict

STEP = "PREPROCESS"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])


if __name__ == "__main__":
    import argparse
    import logging
    from datetime import date, timedelta
    from multiprocessing import Pool, cpu_count

    import boto3
    import preprocess_utils as putils
    import utils
    from dateutil.relativedelta import relativedelta
    from numpy import array_split, cumsum, log, vectorize, where
    from pandas import DataFrame, concat, cut, merge, read_csv, to_datetime
    from pyarrow.parquet import ParquetDataset
    from s3fs import S3FileSystem

    # from pandas import concat, cut, merge, read_csv, read_parquet, to_datetime, DataFrame

    def preprocess(parameters: Dict[Any, Any]) -> DataFrame:
        """Preprocess function to use in a multiprocess way."""
        prep_path = parameters.get("prep_path")
        logger = parameters.get("logger")
        config_gen = parameters.get("config_gen")
        config_step = parameters.get("config_step")
        args__use_type = parameters.get("args.use_type")
        args__s3_bucket = parameters.get("args.s3_bucket")
        args__s3_path = parameters.get("args.s3_path")
        args__model_type = parameters.get("args.model_type")
        args__iata_pct = parameters.get("args.iata_pct")
        args__str_execution_date = parameters.get("args.str_execution_date")
        args__is_last_date = parameters.get("args.is_last_date")
        args__month_threshold_split = parameters.get("args.month_threshold_split")
        target_column = parameters.get("target_column")
        target_mapping = parameters.get("target_mapping")
        year = parameters.get("year")
        month = parameters.get("month")
        day = parameters.get("day")
        output_object_feat = parameters.get("output_object_feat")
        output_object_num = parameters.get("output_object_num")
        output_iata_pct_blv = parameters.get("output_iata_pct_blv")
        output_iata_pct_bl = parameters.get("output_iata_pct_bl")

        logger.info("userlog: Read date path %s.", prep_path)
        data = (
            ParquetDataset(path_or_paths=prep_path, filesystem=S3FileSystem())
            .read_pandas()
            .to_pandas()
        )
        # data = read_parquet(prep_path)
        # Code
        data[config_gen.get("DEPARTURE_TIME_NAME")] = to_datetime(
            data[config_gen.get("DEPARTURE_TIME_NAME")], format="%Y-%m-%d %H:%M:%S"
        )
        data[config_gen.get("WEEKNUM_NAME")] = (
            data[config_gen.get("DEPARTURE_TIME_NAME")]
            .dt.isocalendar()
            .week.astype(int)
        )
        data[config_gen.get("MONTH_NAME")] = data[
            config_gen.get("DEPARTURE_TIME_NAME")
        ].dt.month.astype(int)
        data[config_gen.get("YEAR_NAME")] = data[
            config_gen.get("DEPARTURE_TIME_NAME")
        ].dt.year.astype(int)
        data[config_gen.get("TIME_FLIGHT_NAME")] = cut(
            data[config_gen.get("DEPARTURE_TIME_NAME")].dt.hour,
            bins=config_step.get("TIME_FLIGHT_VALUES").get("bins"),
            labels=config_step.get("TIME_FLIGHT_VALUES").get("labels"),
            include_lowest=True,
        )
        data[config_gen.get("TIME_FLIGHT_NAME")] = data[
            config_gen.get("TIME_FLIGHT_NAME")
        ].astype(object)
        data[config_gen.get("SCALES_NAME")] = (
            data[config_gen.get("ITINERARY_NAME")].str.contains("-").astype(int)
        )
        data[config_gen.get("FLIGHT_ITINERARY_NAME")] = (
            data[config_gen.get("ORIGIN_CITY_NAME")]
            + data[config_gen.get("DESTINATION_CITY_NAME")]
        )
        data[config_gen.get("PAYMENT_NAME")] = data[
            config_gen.get("PAYMENT_NAME")
        ].astype(float)
        # IATA PCT
        if args__use_type in ["predict", "predict-oneshot"]:
            iata_pct_path, _, _, _ = utils.get_path_to_read_and_date(
                read_last_date=True,
                bucket=args__s3_bucket,
                key=f"{args__s3_path}/200_preprocess_step/train/models",
                partition_date=args__str_execution_date.replace("-", ""),
            )
            df_iata_pct = read_csv(
                f"{iata_pct_path}/{config_step.get('IATA_FILENAME')}.csv"
            )
            data = merge(
                left=data,
                right=df_iata_pct,
                how="left",
                on=config_gen.get("FLIGHT_ITINERARY_NAME"),
            )
        else:
            iata_pct_filename = (
                f"s3://{args__s3_bucket}/{args__s3_path}/200_preprocess_step/"
                f"{args__use_type}/models/{args__str_execution_date.replace('-', '')}/"
                f"{config_step.get('IATA_FILENAME')}.csv"
            )
            df_iata_pct = (
                data.groupby([config_gen.get("FLIGHT_ITINERARY_NAME"), target_column])
                .size()
                .reset_index()
                .pivot(
                    index=config_gen.get("FLIGHT_ITINERARY_NAME"),
                    columns=target_column,
                    values=0,
                )
            )
            df_iata_pct["total"] = df_iata_pct.sum(axis=1)
            df_iata_pct["pct_total"] = df_iata_pct["total"] / df_iata_pct["total"].sum()
            df_iata_pct = df_iata_pct.sort_values("total", ascending=False)
            df_iata_pct["cumsum"] = 100 * cumsum(df_iata_pct["pct_total"])
            df_iata_pct = df_iata_pct[
                df_iata_pct["cumsum"] <= float(args__iata_pct)
            ].fillna(0)
            journey_labels = data[target_column].unique()
            for journey_label in journey_labels:
                df_iata_pct[f"feat_{journey_label.lower()}_iata_pct"] = (
                    df_iata_pct[journey_label] / df_iata_pct["total"]
                )
                df_iata_pct = df_iata_pct.drop(columns=[journey_label])
            df_iata_pct = df_iata_pct.drop(columns=["total", "pct_total", "cumsum"])
            df_iata_pct = df_iata_pct.reset_index()
            df_iata_pct.to_csv(iata_pct_filename, index=False)
            data = merge(
                left=data,
                right=df_iata_pct,
                how="left",
                on=config_gen.get("FLIGHT_ITINERARY_NAME"),
            )
        # CLOSE VACATIONS
        for country_code, iatas in config_step.get("BOARDING_AIRPORT_MAPPING").items():
            data[config_gen.get("BOARDPOINT_COUNTRY_CODE_NAME")] = where(
                data[config_gen.get("BOARDING_AIRPORT_NAME")].isin(iatas),
                country_code,
                data[config_gen.get("BOARDPOINT_COUNTRY_CODE_NAME")],
            )
        vymd_to_array_date = vectorize(putils.ymd_to_array_date)
        vdays_to_array_timedelta = vectorize(putils.days_to_array_timedelta)
        vget_holidays = vectorize(putils.get_holidays)
        vconcatenate_arrays_pd = vectorize(putils.concatenate_arrays_pd)
        data["holidays"] = vget_holidays(
            boarding_country_code=data[config_gen.get("BOARDPOINT_COUNTRY_CODE_NAME")],
            offpoint_country_code=data[config_gen.get("OFFPOINT_COUNTRY_CODE_NAME")],
            year=data[config_gen.get("DEPARTURE_TIME_NAME")].dt.year,
        )
        data["array_date"] = vymd_to_array_date(
            data[config_gen.get("DEPARTURE_TIME_NAME")].dt.year,
            data[config_gen.get("DEPARTURE_TIME_NAME")].dt.month,
            data[config_gen.get("DEPARTURE_TIME_NAME")].dt.day,
        )
        data["has_holidays"] = data["holidays"].str.len() != 1
        dataframe_aux1 = data[data["has_holidays"] == False]
        dataframe_aux1[config_gen.get("CLOSE_HOLIDAYS_NAME")] = 0
        # Cuidado con el orden de las ejecuciones. El correcto es el servido.
        dataframe_aux2 = data[data["has_holidays"] == True]
        dataframe_aux3 = dataframe_aux2[
            dataframe_aux2[config_gen.get("DAYS_IN_DESTINATION_NAME")].isnull()
        ]
        dataframe_aux2 = dataframe_aux2[
            dataframe_aux2[config_gen.get("DAYS_IN_DESTINATION_NAME")].notnull()
        ]
        dataframe_aux3[config_gen.get("CLOSE_HOLIDAYS_NAME")] = (
            dataframe_aux3["holidays"] - dataframe_aux3["array_date"]
        )
        dataframe_aux3[config_gen.get("CLOSE_HOLIDAYS_NAME")] = dataframe_aux3[
            config_gen.get("CLOSE_HOLIDAYS_NAME")
        ].apply(
            lambda _: int(
                (abs(_) <= timedelta(days=config_step.get("VACATION_RANGE_DAYS"))).any()
            )
        )
        dataframe_aux2[
            f"{config_gen.get('DAYS_IN_DESTINATION_NAME')}_arr"
        ] = vdays_to_array_timedelta(
            days=dataframe_aux2[config_gen.get("DAYS_IN_DESTINATION_NAME")]
        )
        dataframe_aux2["array_date_aux"] = (
            dataframe_aux2["array_date"]
            + dataframe_aux2[f"{config_gen.get('DAYS_IN_DESTINATION_NAME')}_arr"]
        )
        dataframe_aux2["array_date_aux_p1"] = dataframe_aux2[
            "array_date_aux"
        ] + vdays_to_array_timedelta(days=[1.0] * dataframe_aux2.shape[0])
        dataframe_aux2["array_date_m1"] = dataframe_aux2[
            "array_date"
        ] - vdays_to_array_timedelta(days=[1.0] * dataframe_aux2.shape[0])
        dataframe_aux2["holidays_aux"] = vconcatenate_arrays_pd(
            dataframe_aux2["holidays"].values,
            dataframe_aux2["array_date_m1"].values,
            dataframe_aux2["array_date_aux_p1"].values,
        )
        dataframe_aux2[config_gen.get("CLOSE_HOLIDAYS_NAME")] = dataframe_aux2[
            "holidays_aux"
        ].apply(
            lambda l: int(bool(list(filter(lambda e: l[-2] <= e <= l[-1], l[:-2]))))
        )
        dataframe_aux1 = dataframe_aux1.drop(columns=["has_holidays"])
        dataframe_aux2 = dataframe_aux2.drop(
            columns=[
                "array_date_aux",
                "array_date_aux_p1",
                "array_date_m1",
                "holidays_aux",
                "has_holidays",
            ]
        )
        dataframe_aux3 = dataframe_aux3.drop(columns=["has_holidays"])
        data = concat([dataframe_aux1, dataframe_aux2, dataframe_aux3])
        dataframe_aux1 = data[data[config_gen.get("CLOSE_HOLIDAYS_NAME")] == 0]
        dataframe_aux2 = dataframe_aux1[
            dataframe_aux1[config_gen.get("DAYS_IN_DESTINATION_NAME")].isnull()
        ]
        dataframe_aux1 = dataframe_aux1[
            dataframe_aux1[config_gen.get("DAYS_IN_DESTINATION_NAME")].notnull()
        ]
        dataframe_aux3 = data[data[config_gen.get("CLOSE_HOLIDAYS_NAME")] != 0]
        dataframe_aux1[
            f"{config_gen.get('DAYS_IN_DESTINATION_NAME')}_arr"
        ] = vdays_to_array_timedelta(
            days=dataframe_aux1[config_gen.get("DAYS_IN_DESTINATION_NAME")]
        )
        dataframe_aux1["array_date_aux"] = (
            dataframe_aux1["array_date"]
            + dataframe_aux1[f"{config_gen.get('DAYS_IN_DESTINATION_NAME')}_arr"]
        )
        dataframe_aux1["array_date_aux_p1"] = dataframe_aux1[
            "array_date_aux"
        ] + vdays_to_array_timedelta(days=[1.0] * dataframe_aux1.shape[0])
        dataframe_aux1["array_date_m1"] = dataframe_aux1[
            "array_date"
        ] - vdays_to_array_timedelta(days=[1.0] * dataframe_aux1.shape[0])
        dataframe_aux1 = dataframe_aux1.explode(
            column=["array_date_aux_p1", "array_date_m1"]
        )
        dataframe_aux1["array_date_aux_p1"] = to_datetime(
            dataframe_aux1["array_date_aux_p1"], format="%Y-%m-%d"
        )
        dataframe_aux1["array_date_m1"] = to_datetime(
            dataframe_aux1["array_date_m1"], format="%Y-%m-%d"
        )
        dataframe_aux1[config_gen.get("CLOSE_HOLIDAYS_NAME")] = (
            dataframe_aux1["array_date_aux_p1"].dt.year
            != dataframe_aux1["array_date_m1"].dt.year
        ).astype(int)
        dataframe_aux1 = dataframe_aux1.drop(
            columns=["array_date_aux_p1", "array_date_m1", "array_date_aux"]
        )
        data = concat([dataframe_aux1, dataframe_aux2, dataframe_aux3])
        data = data.drop(
            columns=[
                "holidays",
                "array_date",
                f"{config_gen.get('DAYS_IN_DESTINATION_NAME')}_arr",
            ]
        )
        data[config_gen.get("DAYS_IN_DESTINATION_NAME")] = log(
            data[config_gen.get("DAYS_IN_DESTINATION_NAME")] + 1
        )
        data[config_gen.get("PAYMENT_NAME")] = log(
            data[config_gen.get("PAYMENT_NAME")] + 1
        )
        data[config_gen.get("SEATS_PAYMENT_NAME")] = log(
            data[config_gen.get("SEATS_PAYMENT_NAME")].astype(float) + 1
        )
        data[config_gen.get("BAGS_PAYMENT_NAME")] = log(
            data[config_gen.get("BAGS_PAYMENT_NAME")].astype(float) + 1
        )
        data[config_gen.get("COMPANION_NUMBER_NAME")] = (
            data[config_gen.get("COMPANION_NUMBER_NAME")] - 1
        )
        data[config_gen.get("CAT_COMPANION_NUMBER_NAME")] = data[
            config_gen.get("COMPANION_NUMBER_NAME")
        ]
        data[config_gen.get("BOOL_COMPANION_NUMBER_NAME")] = (
            data[config_gen.get("COMPANION_NUMBER_NAME")].astype(bool).astype(int)
        )
        data[config_gen.get("BAGS_NUMBER_NAME")] = (
            data[config_gen.get("BAGS_NUMBER_NAME")]
            .astype(int)
            .astype(bool)
            .astype(int)
        )
        data[config_gen.get("SEATS_NUMBER_NAME")] = (
            data[config_gen.get("SEATS_NUMBER_NAME")]
            .astype(int)
            .astype(bool)
            .astype(int)
        )
        data[config_gen.get("BOOL_DESTINATION_TIMES_NAME")] = (
            data[config_gen.get("DESTINATION_TIMES_NAME")].astype(bool).astype(int)
        )
        data[config_gen.get("BOOL_DESTINATION_TIMES_LQ_NAME")] = (
            data[config_gen.get("DESTINATION_TIMES_LQ_NAME")].astype(bool).astype(int)
        )
        data[config_gen.get("FIRST_TRAVEL_NAME")] = (
            data[config_gen.get("TRAVEL_TIMES_NAME")]
            .astype(bool)
            .apply(lambda _: not _)
            .astype(int)
        )
        info = {
            config_gen.get("SCALES_NAME"): config_step.get("SCALES_VALUES"),
            config_gen.get("CLOSE_HOLIDAYS_NAME"): config_step.get(
                "CLOSE_HOLIDAYS_VALUES"
            ),
            config_gen.get("NON_WORKING_MAILS_NAME"): config_step.get(
                "NON_WORKING_MAILS_VALUES"
            ),
            config_gen.get("CAT_COMPANION_NUMBER_NAME"): config_step.get(
                "CAT_COMPANION_NUMBER_VALUES"
            ),
            config_gen.get("BOOL_COMPANION_NUMBER_NAME"): config_step.get(
                "BOOL_COMPANION_NUMBER_VALUES"
            ),
            config_gen.get("RESIDENT_NAME"): config_step.get("RESIDENT_VALUES"),
            config_gen.get("AVIOS_NAME"): config_step.get("AVIOS_VALUES"),
            config_gen.get("BAGS_NUMBER_NAME"): config_step.get("BAGS_NUMBER_VALUES"),
            config_gen.get("SEATS_NUMBER_NAME"): config_step.get("SEATS_NUMBER_VALUES"),
            config_gen.get("BOOL_DESTINATION_TIMES_NAME"): config_step.get(
                "BOOL_DESTINATION_TIMES_VALUES"
            ),
            config_gen.get("BOOL_DESTINATION_TIMES_LQ_NAME"): config_step.get(
                "BOOL_DESTINATION_TIMES_VALUES"
            ),
            config_gen.get("FIRST_TRAVEL_NAME"): config_step.get("FIRST_TRAVEL_VALUES"),
        }
        for col, mapping in info.items():
            data[col] = data[col].map(arg=putils.CustomDict(mapping))

        data[target_column] = data[target_column].map(target_mapping)
        data[target_column] = data[target_column].astype(object)
        data[config_gen.get("PCT_REP_VS_TRAVELS_NAME")] = (
            data[config_gen.get("DESTINATION_TIMES_NAME")] + 1
        ) / (data[config_gen.get("TRAVEL_TIMES_NAME")] + 1)
        data[config_gen.get("PCT_REP3M_VS_REP_NAME")] = (
            data[config_gen.get("DESTINATION_TIMES_LQ_NAME")] + 1
        ) / (data[config_gen.get("DESTINATION_TIMES_NAME")] + 1)
        output_feat = (
            [
                *config_gen.get("KEY_COLUMNS"),
                *[config_gen.get("DEPARTURE_TIME_NAME")],
                *output_object_feat,
                *output_object_num,
            ]
            + (output_iata_pct_blv if args__model_type == "blv" else output_iata_pct_bl)
            + [target_column]
        )
        data = data[output_feat]
        threshold_split = int(
            str(
                date(int(year), int(month), int(day))
                - relativedelta(months=int(args__month_threshold_split))
            ).replace("-", "")
        )
        data = putils.preprocess_pipeline(
            data=data,
            s3_bucket=args__s3_bucket,
            s3_path=args__s3_path,
            use_type=args__use_type,
            config=config_gen,
            config_step=config_step,
            target_column=target_column,
            execution_date=f"{year}{month}{day}",
            is_last_date=bool(int(args__is_last_date)),
            logger=logger,
            threshold_split=threshold_split,
        )
        return data

    SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
    SAGEMAKER_LOGGER.setLevel(logging.INFO)
    SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())

    CONFIG = utils.read_config_data()
    CONFIG_STEP = CONFIG.get(f"{STEP}_STEP")
    CONFIG_GEN = CONFIG.get("GENERAL")

    OUTPUT_OBJECT_FEAT = [
        CONFIG_GEN.get("SCALES_NAME"),
        CONFIG_GEN.get("TIME_FLIGHT_NAME"),
        CONFIG_GEN.get("BOARDING_AIRPORT_NAME"),
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
            parser.add_argument(
                "--use_type", type=str, choices=["predict", "train", "predict-oneshot"]
            )
            parser.add_argument("--iata_pct", type=str)
            parser.add_argument("--month_threshold_split", type=str)
            parser.add_argument("--machines", type=str, default="1::1")

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
        target_mapping = (
            CONFIG_GEN.get("BLV_MAPPING")
            if args.model_type == "blv"
            else CONFIG_GEN.get("BL_MAPPING")
        )
        # READ DATA
        if args.use_type == "predict":
            read_step_name = "110_etl_prediction_step"
        else:
            read_step_name = "100_etl_step"
        u_type = args.use_type.split("-")[0]
        preprocess_path, year, month, day = utils.get_path_to_read_and_date(
            read_last_date=bool(int(args.is_last_date)),
            bucket=args.s3_bucket,
            key=f"{args.s3_path}/{read_step_name}/{u_type}",
            partition_date=args.str_execution_date.replace("-", ""),
        )
        preprocess_parameters = {
            "logger": SAGEMAKER_LOGGER,
            "config_gen": CONFIG_GEN,
            "config_step": CONFIG_STEP,
            "args.use_type": args.use_type,
            "args.s3_bucket": args.s3_bucket,
            "args.s3_path": args.s3_path,
            "args.model_type": args.model_type,
            "args.iata_pct": args.iata_pct,
            "args.str_execution_date": args.str_execution_date,
            "args.is_last_date": args.is_last_date,
            "args.month_threshold_split": args.month_threshold_split,
            "target_column": target_column,
            "target_mapping": target_mapping,
            "year": year,
            "month": month,
            "day": day,
            "output_object_feat": OUTPUT_OBJECT_FEAT,
            "output_object_num": OUTPUT_OBJECT_NUM,
            "output_iata_pct_blv": OUTPUT_IATA_PCT_BLV,
            "output_iata_pct_bl": OUTPUT_IATA_PCT_BL,
        }
        if args.use_type == "train":
            preprocess_parameters["prep_path"] = preprocess_path
            dataframe = preprocess(parameters=preprocess_parameters)
        else:
            s3_resource = boto3.resource("s3")
            key_object = preprocess_path.split("/", maxsplit=3)[-1]
            s3_keys = [
                item.key
                for item in s3_resource.Bucket(args.s3_bucket).objects.filter(
                    Prefix=key_object
                )
                if item.key.endswith(".parquet")
            ]
            preprocess_paths = [f"s3://{args.s3_bucket}/{key}" for key in s3_keys]
            num_machines, n_machine = args.machines.split("::")
            preprocess_paths = array_split(
                ary=preprocess_paths, indices_or_sections=int(num_machines)
            )[int(n_machine) - 1].tolist()
            preprocess_parameters_list = [
                {"prep_path": prep_path.tolist(), **preprocess_parameters}
                for prep_path in array_split(
                    ary=preprocess_paths,
                    indices_or_sections=int(len(preprocess_paths) / 1),
                )
            ]
            with Pool(cpu_count()) as pool:
                SAGEMAKER_LOGGER.info(f"userlog: There are {cpu_count()} cores executing in parallel.")
                results = pool.map(preprocess, preprocess_parameters_list)
                dataframe = concat(results, ignore_index=True)
                pool.close()

        # Save data
        save_path = f"s3://{args.s3_bucket}/{args.s3_path}/200_preprocess_step/{u_type}/data/{year}{month}{day}"
        SAGEMAKER_LOGGER.info(
            "userlog: Saving information for preprocess step in %s.", save_path
        )
        if args.use_type == "train":
            complete_save_path = f"{save_path}/preprocess_data.csv"
        else:
            machines_suffix = args.machines.replace("::", "_")
            complete_save_path = f"{save_path}/preprocess_data_{machines_suffix}.csv"
        dataframe.to_csv(complete_save_path, index=False)

    main()
