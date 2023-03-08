"""
ETL utils for BLV model
"""
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.window import Window


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
    return date_time


def avoid_head_rows(dataframe: DataFrame) -> DataFrame:
    """Avoid head rows in data."""
    dataframe = dataframe.where(
        (f.col("cid_analytical") != "cid_analytical")
        | (f.col("cid_analytical").isNull())
    )
    return dataframe


def filter_positive_gross_revenue(dataframe: DataFrame) -> DataFrame:
    """This function filters positive gross revenue and normalize name."""
    dataframe = dataframe.where((f.col("gross_revenue_eur") > 0))
    dataframe = dataframe.withColumnRenamed("gross_revenue_eur", "gross_revenue")
    return dataframe


def create_null_field(column_name: str, null_format: str) -> object:
    """This function change a value by null.

    :param column_name: Column name with value to change.
    :param null_format: Value to change inside column.
    :return: Column with null values instead of null_format.
    """
    return f.when(f.col(column_name).startswith(null_format), f.lit(None)).otherwise(f.col(column_name))


def add_window_feature(
    dataframe: DataFrame,
    keys: List[str],
    order_cols: List[str],
    func: f,
    col_name: str,
    ascending: Optional[List[bool]],
) -> DataFrame:
    """TODO: Add docstring"""
    if ascending is None:
        ascending = [True] * len(order_cols)
    else:
        pass
    order_cols = [
        f.col(oc).asc() if is_asc else f.col(oc).desc()
        for oc, is_asc in zip(order_cols, ascending)
    ]
    window = (
        Window.partitionBy(*keys)
        .orderBy(*order_cols)
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    dataframe = dataframe.withColumn(col_name, func.over(window))
    return dataframe
