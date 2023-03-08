"""
Plot utils for BLV model
"""
from io import BytesIO

from boto3 import resource
from matplotlib import pyplot as plt
from numpy import asarray
from numpy import sum as npsum
from scikitplot.metrics import plot_ks_statistic, plot_precision_recall, plot_roc
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pandas import DataFrame
from typing import List, Tuple, Union

S3_RESOURCE = resource("s3")


def generate_confusion_matrix(
    model: Union[RandomForestClassifier, CatBoostClassifier, BalancedRandomForestClassifier, LGBMClassifier],
    train_set: DataFrame,
    val_set: DataFrame,
    test_set: DataFrame,
    target_col: str,
    feat_col: List[str],
    s3_bucket_name: str,
    s3_path: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """Generate a confussion matrix chart for train, validation and test set.

    :param model: Trained model to predict data.
    :param train_set: Train dataset.
    :param val_set: Validation dataset.
    :param test_set: Test dataset.
    :param target_col: Target column.
    :param feat_col: Features used to predict.
    :param s3_bucket_name: s3 bucket to save chart.
    :param s3_path: s3 path to save chart.
    :return: Three tuples with the values of the confusion mastrix for each set.
    """
    group_names = ["TN", "FP", "FN", "TP"]
    cf_matrix_train = confusion_matrix(
        train_set[target_col], model.predict(train_set[feat_col])
    )
    cf_matrix_validation = confusion_matrix(
        val_set[target_col], model.predict(val_set[feat_col])
    )
    cf_matrix_test = confusion_matrix(
        test_set[target_col], model.predict(test_set[feat_col])
    )

    group_counts_train = [f"{value:,}" for value in cf_matrix_train.flatten()]
    group_counts_validation = [f"{value:,}" for value in cf_matrix_validation.flatten()]
    group_counts_test = [f"{value:,}" for value in cf_matrix_test.flatten()]

    group_percentages_train = [
        f"{value:.2%}" for value in cf_matrix_train.flatten() / npsum(cf_matrix_train)
    ]
    group_percentages_validation = [
        f"{value:.2%}"
        for value in cf_matrix_validation.flatten() / npsum(cf_matrix_validation)
    ]
    group_percentages_test = [
        f"{value:.2%}" for value in cf_matrix_test.flatten() / npsum(cf_matrix_test)
    ]

    labels_train = [
        f"{v1}\n{v2} ({v3})"
        for v1, v2, v3 in zip(group_names, group_counts_train, group_percentages_train)
    ]
    labels_train = asarray(labels_train).reshape(2, 2)

    labels_validation = [
        f"{v1}\n{v2} ({v3})"
        for v1, v2, v3 in zip(
            group_names, group_counts_validation, group_percentages_validation
        )
    ]
    labels_validation = asarray(labels_validation).reshape(2, 2)

    labels_test = [
        f"{v1}\n{v2} ({v3})"
        for v1, v2, v3 in zip(group_names, group_counts_test, group_percentages_test)
    ]
    labels_test = asarray(labels_test).reshape(2, 2)

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))
    heatmap(
        cf_matrix_train,
        ax=axes[0],
        annot=labels_train,
        fmt="",
        cmap="Blues",
        cbar=False,
        annot_kws={"fontsize": "x-large", "weight": "bold"},
    )
    heatmap(
        cf_matrix_validation,
        ax=axes[1],
        annot=labels_validation,
        fmt="",
        cmap="Blues",
        cbar=False,
        annot_kws={"fontsize": "x-large", "weight": "bold"},
    )
    heatmap(
        cf_matrix_test,
        ax=axes[2],
        annot=labels_test,
        fmt="",
        cmap="Blues",
        cbar=False,
        annot_kws={"fontsize": "x-large", "weight": "bold"},
    )
    fig.supxlabel("Predicted", fontsize="x-large")
    fig.supylabel("Actual", x=0.07, fontsize="x-large")
    fig.suptitle("Confusion Matrix", fontsize="xx-large")
    axes[0].set_title("Train")
    axes[1].set_title("Validation")
    axes[2].set_title("Test")

    confusion_matrix_figure = BytesIO()
    plt.savefig(confusion_matrix_figure, format="png")
    confusion_matrix_figure.seek(0)
    bucket = S3_RESOURCE.Bucket(s3_bucket_name)
    bucket.put_object(
        Body=confusion_matrix_figure,
        ContentType="image/png",
        Key=f"{s3_path}/confusion_matrix.png",
    )
    ret = (
        cf_matrix_train.flatten(),
        cf_matrix_validation.flatten(),
        cf_matrix_test.flatten()
    )
    return ret


def generate_metric_plots(
    model: Union[RandomForestClassifier, CatBoostClassifier, BalancedRandomForestClassifier, LGBMClassifier],
    train_set: DataFrame,
    val_set: DataFrame,
    test_set: DataFrame,
    target_col: str,
    feat_col: List[str],
    s3_bucket_name: str,
    s3_path: str,
) -> None:
    """Generate a confussion matrix chart for train, validation and test set.

    :param model: Trained model to predict data.
    :param train_set: Train dataset.
    :param val_set: Validation dataset.
    :param test_set: Test dataset.
    :param target_col: Target column.
    :param feat_col: Features used to predict.
    :param s3_bucket_name: s3 bucket to save chart.
    :param s3_path: s3 path to save chart.
    :return: None.
    """
    _, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 16))
    plot_roc(
        train_set[target_col],
        model.predict_proba(train_set[feat_col]),
        ax=axes[0, 0],
        title="ROC Curves Train",
    )
    plot_roc(
        val_set[target_col],
        model.predict_proba(val_set[feat_col]),
        ax=axes[1, 0],
        title="ROC Curves Validation",
    )
    plot_roc(
        test_set[target_col],
        model.predict_proba(test_set[feat_col]),
        ax=axes[2, 0],
        title="ROC Curves Test",
    )
    plot_precision_recall(
        train_set[target_col],
        model.predict_proba(train_set[feat_col]),
        ax=axes[0, 1],
        title="Precision-Recall Curve Train",
    )
    plot_precision_recall(
        val_set[target_col],
        model.predict_proba(val_set[feat_col]),
        ax=axes[1, 1],
        title="Precision-Recall Curve Validation",
    )
    plot_precision_recall(
        test_set[target_col],
        model.predict_proba(test_set[feat_col]),
        ax=axes[2, 1],
        title="Precision-Recall Curve Test",
    )
    plot_ks_statistic(
        train_set[target_col],
        model.predict_proba(train_set[feat_col]),
        ax=axes[0, 2],
        title="KS Statistic Plot Train",
    )
    plot_ks_statistic(
        val_set[target_col],
        model.predict_proba(val_set[feat_col]),
        ax=axes[1, 2],
        title="KS Statistic Plot Validation",
    )
    plot_ks_statistic(
        test_set[target_col],
        model.predict_proba(test_set[feat_col]),
        ax=axes[2, 2],
        title="KS Statistic Plot Test",
    )

    metric_plots_figure = BytesIO()
    plt.savefig(metric_plots_figure, format="png")
    metric_plots_figure.seek(0)
    bucket = S3_RESOURCE.Bucket(s3_bucket_name)
    bucket.put_object(
        Body=metric_plots_figure,
        ContentType="image/png",
        Key=f"{s3_path}/metric_plots.png",
    )
