"""
Preprocess utils for BLV model
"""
import pickle
from datetime import date, timedelta
from logging import Logger

import boto3
import utils
from category_encoders import TargetEncoder
from holidays import country_holidays
from numpy import array, concatenate, ndarray, where
from pandas import DataFrame
from sklearn.impute import KNNImputer


class CustomDict(dict):
    def __missing__(self, key):
        if "__missing__" in self.keys():
            return self.get("__missing__")
        raise KeyError(
            f"key: {key} doesn't exist. And the dictionary doesn't have a '__missing__' key."
        )


def get_holidays(
    boarding_country_code: str, offpoint_country_code: str, year: int
) -> ndarray:
    """Get holidays given a country code and a year.

    Parameters
    ----------
        boarding_country_code: Boarding country code of 2 characters.
        offpoint_country_code: Offpoint country code of 2 characters.
        year: Year to obtain the holidays.

    Returns
    -------
        An array with the holidays of the given country code and
            year. In case the country code do not exists then the
            result will be an array of one element with None.
    """
    try:
        ret = array(list(country_holidays(boarding_country_code, years=int(year))))
    except Exception:
        try:
            ret = array(list(country_holidays(offpoint_country_code, years=int(year))))
        except Exception:
            ret = [None]
    return ret


def ymd_to_array_date(year: int, month: int, day: int) -> ndarray:
    """Change format of a given year, month and day to an array of the date."""
    return array([date(year, month, day)])


def days_to_array_timedelta(days: float) -> ndarray:
    """Change format of a given year, month and day to an array of the date."""
    return array([timedelta(days=days)])


def concatenate_arrays_pd(*args):
    """Concatenate values in columns with arrays."""
    arr = concatenate(args)
    return arr


def preprocess_pipeline(
    data: DataFrame,
    s3_bucket: str,
    s3_path: str,
    use_type: str,
    config: dict,
    config_step: dict,
    target_column: str,
    execution_date: str,
    is_last_date: bool,
    logger: Logger,
    threshold_split: int,
) -> DataFrame:
    """Transform categorical data in numeric data and apply defined preprocess.

    :param data: Dataframe to preprocess.
    :param s3_bucket: Bucket to save or read the models.
    :param s3_path: Path to save or read the models.
    :param use_type: Type of using this script, train or predict.
    :param config: Configuration dictionary.
    :param config_step: Configuration step dictionary.
    :param target_column: Target column.
    :param execution_date: Execution date.
    :param is_last_date: Look for the last uploaded model.
    :param logger: Logger of the execution.
    :param threshold_split: To create preprocess models over the same perimeter.
    :return: Dataframe processed.
    """

    def imputer() -> None:
        """Null imputer preprocessing."""
        model_name = config_step.get("IMPUTER_MODEL_NAME")
        imputer_columns = list(
            set(data.columns)
            - set(
                config.get("KEY_COLUMNS")
                + [target_column]
                + [config.get("DEPARTURE_TIME_NAME")]
            )
        )
        if use_type == "train":
            subpath = f"{use_type}/models/{execution_date}"
            imputer_ct = KNNImputer(keep_empty_features=True)
            imputer_ct.fit(
                X=data_train[imputer_columns], y=data_train[target_column]
            )
            data[imputer_columns] = imputer_ct.transform(
                X=data[imputer_columns]
            )
            fitted_null_encoder = pickle.dumps(imputer_ct)
            s3_resource.Object(s3_bucket, f"{path}/{subpath}/{model_name}.pkl").put(
                Body=fitted_null_encoder
            )
            logger.info(
                "userlog: Save imputer model in %s.",
                f"{s3_bucket}/{path}/{subpath}/{model_name}.pkl",
            )
        else:
            model_path, _, _, _ = utils.get_path_to_read_and_date(
                read_last_date=is_last_date,
                bucket=s3_bucket,
                key=f"{path}/train/models",
                partition_date=execution_date,
            )
            model_path = model_path.replace("s3://", "").replace(f"{s3_bucket}/", "")
            logger.info(
                "userlog: Read imputer model %s.",
                f"{s3_bucket}/{model_path}/{model_name}.pkl",
            )
            fitted_null_encoder = (
                s3_resource.Bucket(s3_bucket)
                .Object(f"{model_path}/{model_name}.pkl")
                .get()
            )
            imputer_ct = pickle.loads(fitted_null_encoder["Body"].read())
            trained_imputer_columns = imputer_ct.feature_names_in_.tolist()
            data[trained_imputer_columns] = imputer_ct.transform(
                X=data[trained_imputer_columns]
            )

    def categorical_encoder() -> None:
        """Categorical encoder preprocess. TargetEncoding"""
        model_name = config_step.get("ENCODER_MODEL_NAME")
        encoder_columns = [
            config.get("SCALES_NAME"),
            config.get("TIME_FLIGHT_NAME"),
            config.get("BOARDING_AIRPORT_NAME"),
            config.get("CLOSE_HOLIDAYS_NAME"),
            config.get("CAT_COMPANION_NUMBER_NAME"),
            config.get("BOOL_COMPANION_NUMBER_NAME"),
            config.get("BOOL_DESTINATION_TIMES_NAME"),
            config.get("BOOL_DESTINATION_TIMES_LQ_NAME"),
            config.get("FIRST_TRAVEL_NAME"),
            config.get("BAGS_NUMBER_NAME"),
            config.get("NON_WORKING_MAILS_NAME"),
            config.get("HAUL_NAME"),
            config.get("CLASS_NAME"),
            config.get("RESIDENT_NAME"),
            config.get("TIER_NAME"),
            config.get("SALES_CHANNEL_NAME"),
            config.get("WEEKDAY_NAME"),
            config.get("AVIOS_NAME"),
            config.get("SEATS_NUMBER_NAME"),
            config.get("FLEX_FF_NAME"),
            config.get("FARE_FAMILY_NAME"),
            config.get("IS_CORPORATE_NAME"),
        ]
        if use_type == "train":
            subpath = f"{use_type}/models/{execution_date}"
            categorical_ct = TargetEncoder(cols=encoder_columns)
            categorical_ct.fit(
                X=data_train[encoder_columns], y=data_train[target_column]
            )
            data[encoder_columns] = categorical_ct.transform(
                X=data[encoder_columns]
            )
            data_train[encoder_columns] = categorical_ct.transform(
                X=data_train[encoder_columns]
            )
            fitted_categorical_encoder = pickle.dumps(categorical_ct)
            s3_resource.Object(s3_bucket, f"{path}/{subpath}/{model_name}.pkl").put(
                Body=fitted_categorical_encoder
            )
            logger.info(
                "userlog: Save categorical encoder model in %s.",
                f"{s3_bucket}/{path}/{subpath}/{model_name}.pkl",
            )

        else:
            model_path, _, _, _ = utils.get_path_to_read_and_date(
                read_last_date=is_last_date,
                bucket=s3_bucket,
                key=f"{path}/train/models",
                partition_date=execution_date,
            )
            model_path = model_path.replace("s3://", "").replace(f"{s3_bucket}/", "")
            logger.info(
                "userlog: Read categorical encoder model %s.",
                f"{s3_bucket}/{model_path}/{model_name}.pkl",
            )
            fitted_categorical_encoder = (
                s3_resource.Bucket(s3_bucket)
                .Object(f"{model_path}/{model_name}.pkl")
                .get()
            )
            categorical_ct = pickle.loads(fitted_categorical_encoder["Body"].read())
            trained_encoder_columns = categorical_ct.get_feature_names_in()
            data[trained_encoder_columns] = categorical_ct.transform(
                X=data[trained_encoder_columns]
            )

    s3_resource = boto3.resource("s3")
    path = f"{s3_path}/200_preprocess_step"
    if use_type == "train":
        data_train = data.copy(deep=True)
        data_train[config.get("DEPARTURE_TIME_NAME")] = data_train[config.get("DEPARTURE_TIME_NAME")].astype(str)
        data_train["date_creation_pnr_resiber"] = data_train["date_creation_pnr_resiber"].astype(str)
        data_train["date_creation_pnr_resiber"] = where(
            data_train["date_creation_pnr_resiber"].isnull(),
            data_train[config.get("DEPARTURE_TIME_NAME")].str.replace("-", ""),
            data_train["date_creation_pnr_resiber"],
        )
        data_train["date_creation_pnr_resiber"] = (
            data_train["date_creation_pnr_resiber"].str.split(" ").str[0].astype(int)
        )
        data_train = data_train[data_train["date_creation_pnr_resiber"] < threshold_split]

    categorical_encoder()
    imputer()
    return data
