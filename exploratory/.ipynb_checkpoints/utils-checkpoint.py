"""
Utils for the ETL Pipeline
"""
from argparse import Namespace
from logging import Logger
from os import popen
from os.path import join

from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.window import Window


def get_last_s3_partition(s3_dir: str, grep: str = "") -> str:
    """This functions assumes that the last partition is the last one that appears on `aws s3 ls` command
    Parameters
    ----------
    s3_dir: s3 path with churn data ending with '/'
    grep: string to filter path

    Returns
    -------
    complete path to the last partition
    """
    if grep:
        var = popen(f"aws s3 ls {s3_dir} | grep {grep} |tail -n 1 | awk '{{print $2}}'").read()
    else:
        var = popen(f"aws s3 ls {s3_dir} | tail -n 1 | awk '{{print $2}}'").read()
    return join(s3_dir, var.split("/")[0])


def eliminate_time(date_time: str, sep: str) -> str:
    """This function split a datetime string to eliminate time.

    Parameters
    ----------
        date_time: datetime string
        sep: separator between date and time

    Returns
    -------
        String with date
    """
    if not date_time is None:
        return date_time.strip().split(sep)[0]
    else:
        return date_time


def coalesce_pnr_columns(dataframe: DataFrame) -> DataFrame:
    """Coalesce pnr columns to catch clients."""
    eliminateTimeUDF = f.udf(lambda x: eliminate_time(x, " "))

    dataframe = dataframe.withColumn(
        "ticket_sale_date_cast",
        f.to_date(
            f.col("ticket_sale_date"),
            "yyyy-MM-dd"
        ).alias("date_2")
    )
    dataframe = dataframe.withColumn(
        "date_creation_pnr_resiber",
        f.to_date(
            eliminateTimeUDF(f.col("date_creation_pnr_resiber")),
            "yyyy-MM-dd"
        ).alias("date_1")
    )

    dict_columns = {
        "date_creation_pnr_resiber": "ticket_sale_date_cast",
        "pnr_resiber": "pnr_amadeus"
    }
    for col_key, col_value in dict_columns.items():
        dataframe = dataframe.withColumn(col_key, f.coalesce(f.col(col_key), col_value))

    return dataframe


def avoid_head_rows(dataframe: DataFrame) -> DataFrame:
    """Avoid head rows in data."""
    dataframe = dataframe.where(
        (f.col('id_golden_record') != 'id_golden_record') |
        (f.col('id_golden_record').isNull())
    )
    return dataframe


def filter_positive_gross_revenue(dataframe: DataFrame) -> DataFrame:
    """This function filters positive gross revenue and normalize name."""
    dataframe = dataframe.where((f.col("gross_revenue_eur") > 0))
    dataframe = dataframe.withColumnRenamed(
        "gross_revenue_eur", "gross_revenue"
    )
    return dataframe


def filter_true_clients(dataframe: DataFrame) -> DataFrame:
    """Filter True clients to use in the recommender system."""
    dataframe = dataframe.where(
        (f.col('coupon_usage_code').isin(['T', 'N'])) &
        (f.col('revenue_pax_ind') == 'Y')
    )
    dataframe = dataframe.where(f.col("cid_analytical").isNotNull())
    return dataframe


def delete_air_bridge(dataframe: DataFrame) -> DataFrame:
    """Delete the following flights MAD-BCN & BCN-MAD."""
    dataframe = dataframe.withColumn(
        "is_air_bridge",
        f.when(
            (
                    (f.col("origin_city_od") == "MAD") &
                    (f.col("destination_city_od") == "BCN")
            ) |
            (
                    (f.col("origin_city_od") == "BCN") &
                    (f.col("destination_city_od") == "MAD")
            ), f.lit(True)
        ).otherwise(f.lit(False))
    )
    dataframe = dataframe.filter(f.col("is_air_bridge") == False)
    return dataframe


def delete_return_trips(dataframe: DataFrame) -> DataFrame:
    """Delete return trips over the same pnr."""
    win_key = Window.partitionBy(
        "cid_analytical", "pnr_resiber", "date_creation_pnr_resiber"
    )
    dataframe = dataframe.orderBy(
        f.col("cid_analytical").asc(), f.col("pnr_resiber").asc(),
        f.col("date_creation_pnr_resiber").asc(), f.col("loc_dep_time_scheduled").asc()
    )
    dataframe = dataframe.withColumn(
        "first_origin",
        f.first("origin_city_od").over(win_key.orderBy(f.col("loc_dep_time_scheduled").asc()))
    )
    dataframe = dataframe.withColumn(
        "is_return_trip",
        (f.col("first_origin") == f.col("destination_city_od"))
    )
    dataframe = dataframe.filter(f.col("is_return_trip") == False)
    drop_cols = [
        "is_return_trip", "first_origin"
    ]
    dataframe = dataframe.drop(*drop_cols)
    return dataframe


def delete_business_flights(dataframe: DataFrame) -> DataFrame:
    """Delete business flights from column ind_reason_business."""
    dataframe = dataframe.filter(f.col("ind_reason_business") == 0)
    return dataframe


def delete_iberia_spain_iberia_hubs(dataframe: DataFrame) -> DataFrame:
    """Delete Iberia hubs to not take into account return flights to hubs for spain id_golden_record."""
    hubs = ["MAD"]
    dataframe = dataframe.withColumn(
        "is_not_hub", f.when(
            (
                    (f.col("nationality") == "ES") &
                    f.col("destination_city_od").isin(hubs)
            ), f.lit(0)).otherwise(f.lit(1))
    )
    dataframe = dataframe.filter(f.col("is_not_hub") == 1)
    return dataframe


def get_path_to_read_and_date(
        read_last_date: bool, bucket: str, key: str, partition_date: str
):
    """Get path to read (given or last) and the chosen date.

    Parameters
    ----------
        read_last_date: Boolean to read last valid date (True) or given
            date (False).
        bucket: S3 bucket.
        key: S3 key.
        partition_date: String with the execution date
            (could be separated by '=' sign).

    Returns
    -------
        Tuple[
            Path with data,
            year of the read data,
            month of the read data,
            day of the read data
        ]
    """
    if read_last_date:
        path = get_last_s3_partition(s3_dir=f"{bucket}/{key}/")
        # date = path.split("/")[-1].split("=")[-1].replace("-", "")
        date = partition_date.split("/")[-1].split("=")[-1].replace("-", "")
        year, month, day = date[:4], date[4:6], date[6:]
        path = f"s3://{path}"
    else:
        path = f"s3://{bucket}/{key}/{partition_date}"
        date = partition_date.split("=")[-1]
        if "-" in partition_date:
            date = date.split("-")
            year, month, day = date[0], date[1], date[2]
        else:
            year, month, day = date[:4], date[4:6], date[6:]
    return path, year, month, day


class AbstractArguments:

    def __init__(self) -> None:
        """Abstract constructor."""
        self.args = NotImplemented

    def get_arguments(self) -> Namespace:
        """Get dictionary with arguments."""
        return self.args

    def info(self, logger: Logger) -> None:
        """Add to the logger the chosen arguments."""
        for arg_name, arg_value in vars(self.args).items():
            logger.info(f"--> Argument --{arg_name}={arg_value}")
