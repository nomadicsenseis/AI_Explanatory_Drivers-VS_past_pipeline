"""
ETL Step for BLVM.
"""
import argparse
import logging
from datetime import date

import etl_utils as eutils
import utils
from dateutil import relativedelta
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.window import Window

SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())

STEP = "ETL"


class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        super().__init__()
        parser = argparse.ArgumentParser(description=f"Inputs for {STEP} step.")
        parser.add_argument("--s3_bucket", type=str)
        parser.add_argument("--s3_path_read", type=str)
        parser.add_argument("--s3_path", type=str)
        parser.add_argument("--str_execution_date", type=str)
        parser.add_argument("--model_type", choices=["bl", "blv"], type=str)
        parser.add_argument("--is_last_date", type=str, default="1")
        parser.add_argument(
            "--use_type", type=str, choices=["predict", "train", "predict-oneshot"]
        )
        self.args = parser.parse_args()


def main():
    """Main functionality of the script."""

    # DEFINE ARGUMENTS
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()
    # CREATE SPARK SESSION AND PYSPARK VARIABLES
    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    configuration = spark.sparkContext.getConf().getAll()
    for pys_conf in configuration:
        if (pys_conf[0] == "spark.yarn.dist.files") and (pys_conf[1] is not None):
            spark.sparkContext.addFile(pys_conf[1])
        SAGEMAKER_LOGGER.info(
            "userlog: Pyspark configuration %s:%s", pys_conf[0], pys_conf[1]
        )
    config = utils.read_config_data(path=SparkFiles.get(filename="config.yml"))
    config_step = config.get(f"{STEP}_STEP")
    config_gen = config.get("GENERAL")
    output_features_step = [
        config_gen.get("DESTINATION_TIMES_NAME"),
        config_gen.get("TRAVEL_TIMES_NAME"),
        config_gen.get("DESTINATION_TIMES_LQ_NAME"),
        config_gen.get("TRAVEL_TIMES_LQ_NAME"),
        config_gen.get("NON_WORKING_MAILS_NAME"),
        config_gen.get("COMPANION_NUMBER_NAME"),
        config_gen.get("HAUL_NAME"),
        config_gen.get("CLASS_NAME"),
        config_gen.get("RESIDENT_NAME"),
        config_gen.get("TIER_NAME"),
        config_gen.get("SALES_CHANNEL_NAME"),
        config_gen.get("WEEKDAY_NAME"),
        config_gen.get("AVIOS_NAME"),
        config_gen.get("DAYS_IN_DESTINATION_NAME"),
        config_gen.get("DESTINATION_CITY_NAME"),
        config_gen.get("ORIGIN_CITY_NAME"),
        config_gen.get("ITINERARY_NAME"),
        config_gen.get("PAYMENT_NAME"),
        config_gen.get("BOARDPOINT_COUNTRY_CODE_NAME"),
        config_gen.get("OFFPOINT_COUNTRY_CODE_NAME"),
        config_gen.get("DEPARTURE_TIME_NAME"),
        config_gen.get("BAGS_PAYMENT_NAME"),
        config_gen.get("BAGS_NUMBER_NAME"),
        config_gen.get("SEATS_PAYMENT_NAME"),
        config_gen.get("SEATS_NUMBER_NAME"),
        config_gen.get("BOARDING_AIRPORT_NAME"),
        config_gen.get("FLEX_FF_NAME"),
        config_gen.get("FARE_FAMILY_NAME"),
        config_gen.get("IS_CORPORATE_NAME"),
    ]
    target_column = (
        config_gen.get("BLV_TARGET_COL")
        if args.model_type == "blv"
        else config_gen.get("BL_TARGET_COL")
    )
    window_key_cols = (
        Window()
        .partitionBy(*config_gen.get("KEY_COLUMNS"))
        .orderBy(f.col("loc_dep_time").asc())
    )
    window_last_quarter = (
        Window()
        .partitionBy("cid_analytical")
        .orderBy(f.col("loc_dep_time").cast("timestamp").cast("long"))
        .rangeBetween(-7776000, -1)
    )
    window_destination_last_quarter = (
        Window()
        .partitionBy("cid_analytical", "destination_city_od")
        .orderBy(f.col("loc_dep_time").cast("timestamp").cast("long"))
        .rangeBetween(-7776000, -1)
    )
    window_companion = Window.partitionBy(["pnr_resiber", "date_creation_pnr_resiber"])
    # READ DATA
    path_source_data, year, month, day = utils.get_path_to_read_and_date(
        read_last_date=bool(int(args.is_last_date)),
        bucket=args.s3_bucket,
        key=args.s3_path_read,
        partition_date=f"insert_date_ci={args.str_execution_date}",
    )
    SAGEMAKER_LOGGER.info("userlog: Read date path %s.", path_source_data)
    dataframe = spark.read.option("header", "true").csv(path_source_data)
    dataframe_count = dataframe.count()
    SAGEMAKER_LOGGER.info("userlog: Init count %s.", dataframe_count)
    # ETL CODE
    exc_date = date(int(year), int(month), int(day))
    dataframe = dataframe.withColumn(
        "date_creation_pnr_resiber",
        eutils.create_null_field("date_creation_pnr_resiber", "0001-01-01"),
    )
    dataframe = eutils.avoid_head_rows(dataframe=dataframe)
    dataframe = dataframe.filter(f.col("ticketing_carrier") == "075")
    dataframe = dataframe.where(f.col("cid_analytical").isNotNull())
    dataframe = dataframe.na.fill(value="UNKNOWN", subset=["nps_journey_reason"])
    dataframe = dataframe.withColumn(
        config_gen.get("BLV_TARGET_COL"),
        f.when(
            f.col("nps_journey_reason").isin(["Short break", "Holiday"]),
            f.lit("Leisure"),
        ).otherwise(
            f.when(
                f.col("nps_journey_reason").isin(["Visiting friends or family"]),
                f.lit("VFF"),
            ).otherwise(
                f.when(
                    f.col("nps_journey_reason") == "UNKNOWN",
                    f.col("nps_journey_reason"),
                ).otherwise(f.lit("Business"))
            )
        ),
    )
    dataframe = dataframe.withColumn(
        config_gen.get("BL_TARGET_COL"),
        f.when(
            f.col(config_gen.get("BLV_TARGET_COL")) == "VFF", f.lit("Leisure")
        ).otherwise(f.col(config_gen.get("BLV_TARGET_COL"))),
    )

    # keep first flight, or delete return flights.
    dataframe = dataframe.withColumn("rn", f.row_number().over(window_key_cols))
    dataframe = dataframe.filter(f.col("rn") == 1).drop("rn")

    # Number of times that an id has repeated the destination
    dataframe = eutils.add_window_feature(
        dataframe=dataframe,
        keys=["cid_analytical", "destination_city_od"],
        order_cols=["loc_dep_time"],
        ascending=[True],
        func=f.count(target_column),
        col_name=config_gen.get("DESTINATION_TIMES_NAME"),
    )

    # Number of times that an id has travel
    dataframe = eutils.add_window_feature(
        dataframe=dataframe,
        keys=["cid_analytical"],
        order_cols=["loc_dep_time"],
        ascending=[True],
        func=f.count(target_column),
        col_name=config_gen.get("TRAVEL_TIMES_NAME"),
    )

    # Number of times that an id has repeated the destination in the last quarter
    dataframe = dataframe.withColumn(
        config_gen.get("DESTINATION_TIMES_LQ_NAME"),
        f.count(target_column).over(window_destination_last_quarter),
    )

    # Number of times that an id has travel in the last quarter
    dataframe = dataframe.withColumn(
        config_gen.get("TRAVEL_TIMES_LQ_NAME"),
        f.count(target_column).over(window_last_quarter),
    )

    # Categorize if a row has a non working mail
    working_name = config_gen.get("NON_WORKING_MAILS_NAME")
    dataframe = dataframe.withColumn(
        working_name, f.split("email_operative", "@").getItem(1)
    )
    dataframe = dataframe.withColumn(
        working_name, f.split(working_name, r"\.").getItem(0)
    )
    dataframe = dataframe.withColumn(
        working_name,
        f.when(
            f.col(working_name).isin(config_step.get("NON_WORKING_MAILS_VALUES")),
            f.lit(0),
        ).otherwise(f.lit(1)),
    )

    # Categorize if the id take a flight with someone else
    dataframe = dataframe.withColumn(
        config_gen.get("COMPANION_NUMBER_NAME"),
        f.approx_count_distinct("cid_analytical").over(window_companion),
    )

    # Transform haul
    dataframe = dataframe.withColumn(
        config_gen.get("HAUL_NAME"),
        f.when(f.col("haul") == "DO", f.lit("SH")).otherwise(f.col("haul")),
    )

    # Transform class
    dataframe = dataframe.withColumn(
        config_gen.get("CLASS_NAME"),
        f.when(
            f.col("sold_class_code").isin(config_step.get("BUSINESS_CLASS_VALUES")),
            f.lit("Business"),
        ).otherwise(
            f.when(
                f.col("sold_class_code").isin(config_step.get("PREMIUM_CLASS_VALUES")),
                f.lit("Premium"),
            ).otherwise(f.lit("Economy"))
        ),
    )

    # Transform tier
    dataframe = dataframe.na.fill(value="No FF", subset=["ff_tier"])
    dataframe = dataframe.withColumn(
        config_gen.get("TIER_NAME"),
        f.when(
            f.col("ff_tier").isin(["Singular", "Infinita Prime", "Infinita"]),
            f.lit("Platino"),
        ).otherwise(f.col("ff_tier")),
    )

    # Categorize if the id corresponds with a resident
    dataframe = dataframe.na.fill(value="UNKNOWN", subset=["pax_type_seg"])
    dataframe = dataframe.withColumn(
        config_gen.get("RESIDENT_NAME"),
        f.when(f.col("pax_type_seg").contains("RESIDENTE"), f.lit(1)).otherwise(
            f.lit(0)
        ),
    )

    # Get the weekday name of the flight
    dataframe = dataframe.withColumn(
        config_gen.get("WEEKDAY_NAME"), f.date_format("loc_dep_time", "E")
    )

    # Transform sales channel
    dataframe = dataframe.withColumn(
        config_gen.get("SALES_CHANNEL_NAME"),
        f.when(
            f.col("ind_direct_sale") == 1, f.lit("DIR")
        ).otherwise(
            f.lit("IND")
        )
    )

    # Transform avios
    dataframe = dataframe.withColumn(
        config_gen.get("AVIOS_NAME"),
        f.when(f.col("revenue_avios") > 0, f.lit(1)).otherwise(f.lit(0)),
    )

    # Transform number of hours in destination
    dataframe = dataframe.withColumn(
        config_gen.get("DAYS_IN_DESTINATION_NAME"),
        f.col("num_hours_in_destination") / 24,
    )

    # Group fare_family
    dataframe = dataframe.withColumn(
        config_gen.get("FARE_FAMILY_NAME"),
        f.when(
            f.col("grouped_family_name").isNull(), f.lit("UNKNOWN")
        ).otherwise(f.col("grouped_family_name"))
    )

    # Is fare_family flex
    dataframe = dataframe.withColumn(
        config_gen.get("FLEX_FF_NAME"),
        f.when(
            f.upper("fare_family").contains("FLEX"), f.lit("Flex FF")
        ).otherwise(f.lit("No flex FF"))
    )

    # Is corporate
    dataframe = dataframe.withColumn(
        config_gen.get("IS_CORPORATE_NAME"),
        f.when(
            f.col("is_corporate") == 1, f.lit("Corporate")
        ).otherwise(f.lit("Non corporate"))
    )

    # Rename columns
    rename_columns = {
        "destination_city_od": config_gen.get("DESTINATION_CITY_NAME"),
        "origin_city_od": config_gen.get("ORIGIN_CITY_NAME"),
        "itinerary_od": config_gen.get("ITINERARY_NAME"),
        "eur_seats": config_gen.get("SEATS_PAYMENT_NAME"),
        "eur_bags": config_gen.get("BAGS_PAYMENT_NAME"),
        "num_seats": config_gen.get("SEATS_NUMBER_NAME"),
        "num_bags": config_gen.get("BAGS_NUMBER_NAME"),
        "total_payment_eur": config_gen.get("PAYMENT_NAME"),
        "loc_dep_time": config_gen.get("DEPARTURE_TIME_NAME"),
        "boardpoint_country_code": config_gen.get("BOARDPOINT_COUNTRY_CODE_NAME"),
        "offpoint_country_code": config_gen.get("OFFPOINT_COUNTRY_CODE_NAME"),
        "boardpoint_airport": config_gen.get("BOARDING_AIRPORT_NAME"),
    }

    # SAVE DATA
    if args.use_type == "predict":
        six_month_earlier = exc_date - relativedelta.relativedelta(months=6)
        dataframe = dataframe.filter(
            (f.col("loc_dep_date") >= six_month_earlier)
        )
    elif args.use_type == "predict-oneshot":
        pass
    else:
        where_condition = ((f.col("coupon_usage_code").isin(["T", "N"])) & (f.col("revenue_pax_ind") == "Y"))
        dataframe = dataframe.where(where_condition)
        dataframe = dataframe.filter(f.col(target_column) != "UNKNOWN")
        two_years_ago = date(int(year) - 2, int(month), int(day))
        dataframe = dataframe.filter((f.col("loc_dep_date") >= two_years_ago))

    for col_name, new_col_name in rename_columns.items():
        dataframe = dataframe.withColumnRenamed(col_name, new_col_name)

    dataframe = dataframe.withColumn(
        "date_creation_pnr_resiber",
        f.regexp_replace("date_creation_pnr_resiber", "-", ""),
    )

    # The are one way to add more columns in this step:
    #     - Calculate a column in the script, add the column name
    #       to the config.yml by a key which has to end with '_NAME' and
    #       add it to 'output_features_step' variable.
    dataframe = dataframe.select(
        *config_gen.get("KEY_COLUMNS"), *output_features_step, target_column
    )
    total_count = dataframe.count()
    solve_null_cols = [config_gen.get("ITINERARY_NAME")]
    null_values = dataframe.select(
        [f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in output_features_step]
    ).collect()[0].asDict()
    for key, value in null_values.items():
        null_pct = round(100 * int(value)/total_count, 5)
        if value:
            SAGEMAKER_LOGGER.warning(f"userlog: Column {key} has {null_pct}% ({value}) of null values.")
            if key in solve_null_cols:
                dataframe = dataframe.na.fill(value="", subset=[key])
            else:
                continue
        else:
            continue

    u_type = args.use_type.split("-")[0]
    save_path = (
        f"s3://{args.s3_bucket}/{args.s3_path}/100_etl_step/{u_type}/{year}{month}{day}"
    )
    SAGEMAKER_LOGGER.info("userlog: Saving information for etl step in %s.", save_path)
    SAGEMAKER_LOGGER.info(
        "userlog: Total data for etl step %s.", str(total_count)
    )
    dataframe.write.format("parquet").option("header", True).mode("overwrite").save(
        save_path
    )


if __name__ == "__main__":
    main()
