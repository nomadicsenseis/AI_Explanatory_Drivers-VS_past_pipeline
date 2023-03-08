"""
Train utils for BLV model
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from optuna.trial import Trial
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

_AVAILABLE_MODELS = [
    "LGBMClassifier",
    "RandomForestClassifier",
    "CatBoostClassifier",
    "ImbalancedRandomForestClassifier",
]


class HPTClassifier:
    """Class to execute an optimization for a given classification model."""

    categorical: List[str] = ["cat", "str", "categorical", "string"]
    integer: List[str] = ["int", "integer"]
    decimal: List[str] = ["float", "decimal"]
    available_metrics: List[str] = ["accuracy", "f1-score"]

    def __init__(
        self,
        clf: Callable,
        params: List[Dict[str, Union[str, List[Union[str, float]]]]],
        train_set: DataFrame,
        validation_set: DataFrame,
        feature_cols: List[str],
        target_col: str,
        opt_metric: Union[str, Callable] = "accuracy",
        **kwargs: Union[str, float],
    ) -> None:
        """Class constructor.

        Parameters
        ----------
            clf: Model in scikit-learn like way to optimize.
            params: Information about the hyperparameters of the clf model. It has
                to satisfy the following format:
                >> params = [
                    {
                        'hp_type': value type for the hp,
                        'hp_name': Name of the hyperparameter,
                        'hp_value': value or values to optimize
                    },
                    {
                        'hp_type': value type for the hp,
                        'hp_name': Name of the hyperparameter,
                        'hp_value': value or values to optimize
                    },
                    ...
                ]
                One dictionary per hyperparameter (hp). 'hp_value' could be a list of choices for
                categorical values, a list of min and max values to optimize the hp or just a fix value
                for the hp.
            train_set: Training data.
            validation_set: Validation data.
            feature_cols: Columns to use as features.
            target_col: Column to use as target.
            opt_metric: Metric to optimize.
        """
        self.__check_param_recipe(params)
        self.clf = clf
        self.params = params
        self.train_set = train_set[feature_cols + [target_col]]
        self.validation_set = validation_set[feature_cols + [target_col]]
        self.feature_cols = feature_cols
        self.target_col = [target_col]
        self.opt_metric = opt_metric
        self.kwargs = kwargs

    @staticmethod
    def __check_param_recipe(
        params: List[Dict[str, Union[str, List[Union[str, float]]]]]
    ) -> None:
        """Checks the format of the given information for the hyperparameters.

        Parameters
        ----------
            params: List of dict with the requiered information. The dicts must have
                the following keys:
                    - 'hp_type': Specify if the hyperparameter is an int, float or str.
                    - 'hp_name': Name of the hyperparameter to be tuned.
                    - 'hp_value': List of values to optimize or value to set for the
                        hyperparameter.
        """
        must_have = ["hp_type", "hp_name", "hp_value"]
        n_bad_formed = 0
        for hyperparameter in params:
            keys = set(hyperparameter.keys()) - set(hyperparameter.keys()).intersection(
                set(must_have)
            )
            if len(keys):
                n_bad_formed += 1
            else:
                continue
        if n_bad_formed:
            raise ValueError(
                f"There are {n_bad_formed} hyperparameters bad-formed. "
                "To consider an hyperparameter the recipe is a list of dictionaries "
                "with 'hp_type', 'hp_name' and 'hp_value as keys'."
            )

    def _get_hyperparametes(self, trial: Trial) -> Dict[str, Union[str, float]]:
        """Transform given hyperparameters in a format accepted for the model.

        Parameters
        ----------
            trial: Trial for the Optuna optimization.

        Returns
        -------
            Dictionary with the variable names and the values to be tested in the
                optimization process.
        """
        hp_categorical = filter(
            lambda element: element["hp_type"] in self.categorical, self.params
        )
        hp_integer = filter(
            lambda element: element["hp_type"] in self.integer, self.params
        )
        hp_float = filter(
            lambda element: element["hp_type"] in self.decimal, self.params
        )

        hp4optimize = {}
        for hyperparamenter in hp_categorical:
            name = hyperparamenter["hp_name"]
            value = hyperparamenter["hp_value"]
            hp4optimize[name] = (
                value
                if not isinstance(value, list)
                else trial.suggest_categorical(name=name, choices=value)
            )
        for hyperparamenter in hp_integer:
            name = hyperparamenter["hp_name"]
            hp4optimize[name] = (
                hyperparamenter["hp_value"]
                if not isinstance(hyperparamenter["hp_value"], list)
                else trial.suggest_int(
                    name=name,
                    low=min(hyperparamenter["hp_value"]),
                    high=max(hyperparamenter["hp_value"]),
                )
            )
        for hyperparamenter in hp_float:
            name = hyperparamenter["hp_name"]
            hp4optimize[name] = (
                hyperparamenter["hp_value"]
                if not isinstance(hyperparamenter["hp_value"], list)
                else trial.suggest_float(
                    name=name,
                    low=min(hyperparamenter["hp_value"]),
                    high=max(hyperparamenter["hp_value"]),
                    log=True,
                )
            )

        return hp4optimize

    def _get_metric(self, pipe) -> float:
        """Get the metric score to optimize.

        Parameters
        ----------
            pipe: Trained model pipeline.

        Returns
        -------
            Metric to optimize.
        """
        if isinstance(self.opt_metric, Callable):
            metric = self.opt_metric(
                self.validation_set[self.target_col],
                pipe.predict(self.validation_set[self.feature_cols]),
            )
        elif self.opt_metric == "accuracy":
            metric = accuracy_score(
                self.validation_set[self.target_col],
                pipe.predict(self.validation_set[self.feature_cols]),
                **self.kwargs,
            )

        elif self.opt_metric == "f1-score":
            metric = f1_score(
                self.validation_set[self.target_col],
                pipe.predict(self.validation_set[self.feature_cols]),
                **self.kwargs,
            )
        else:
            raise ValueError(
                "The given metric is not defined in the class, "
                f"please use an user define metric or one of "
                f"{', '.join(self.available_metrics)}."
            )
        return metric

    def __call__(self, trial: Optional[Trial]) -> float:
        """Use the class as function to perform hpt with Optuna."""
        clfm = self.clf(**self._get_hyperparametes(trial=trial))
        clfm.fit(self.train_set[self.feature_cols], self.train_set[self.target_col])
        metric = self._get_metric(pipe=clfm)

        return metric


def get_model_and_params(
    model_name: str,
) -> Tuple[Callable, List[Dict[str, Optional[Union[List[float], List[str]]]]]]:
    """Get classification model and parameters to optimize.

    :param model_name: Name of the model to use in the optimization.
    :return: Chosen estimator and its hyperparameters.
    """
    if model_name == "LGBMClassifier":
        estimator = LGBMClassifier
        parameters = [
            {"hp_type": "int", "hp_name": "num_iterations", "hp_value": [50, 150]},
            {
                "hp_type": "cat",
                "hp_name": "boosting_type",
                "hp_value": ["gbdt", "dart"],
            },
            {"hp_type": "int", "hp_name": "num_leaves", "hp_value": [10, 50]},
            {"hp_type": "int", "hp_name": "max_depth", "hp_value": [5, 50]},
            {"hp_type": "float", "hp_name": "learning_rate", "hp_value": [1e-2, 1e-1]},
            {"hp_type": "int", "hp_name": "n_estimators", "hp_value": [40, 120]},
            {
                "hp_type": "cat",
                "hp_name": "objective",
                "hp_value": ["binary"],
            },
            {"hp_type": "cat", "hp_name": "class_weight", "hp_value": None},
            {"hp_type": "float", "hp_name": "reg_alpha", "hp_value": [1e-5, 1.0]},
            {"hp_type": "float", "hp_name": "reg_lambda", "hp_value": [1e-5, 1.0]},
            {"hp_type": "float", "hp_name": "min_split_gain", "hp_value": [1e-5, 1.0]},
            {
                "hp_type": "float",
                "hp_name": "min_child_weight",
                "hp_value": [1e-5, 1e-1],
            },
            {"hp_type": "int", "hp_name": "min_child_samples", "hp_value": [10, 100]},
        ]
    elif model_name == "RandomForestClassifier":
        estimator = RandomForestClassifier
        parameters = [
            {"hp_type": "int", "hp_name": "n_estimators", "hp_value": [50, 150]},
            {
                "hp_type": "cat",
                "hp_name": "criterion",
                "hp_value": ["gini", "entropy", "log_loss"],
            },
            {"hp_type": "int", "hp_name": "max_depth", "hp_value": [5, 100]},
            {"hp_type": "int", "hp_name": "min_samples_split", "hp_value": [2, 50]},
            {"hp_type": "int", "hp_name": "min_samples_leaf", "hp_value": [1, 10]},
            {
                "hp_type": "float",
                "hp_name": "min_weight_fraction_leaf",
                "hp_value": [1e-5, 0.5],
            },
            {"hp_type": "cat", "hp_name": "max_features", "hp_value": ["sqrt", "log2"]},
            {
                "hp_type": "cat",
                "hp_name": "class_weight",
                "hp_value": [None, "balanced"],
            },
        ]
    elif model_name == "CatBoostClassifier":
        estimator = CatBoostClassifier
        parameters = [
            {"hp_type": "int", "hp_name": "iterations", "hp_value": [250, 750]},
            {"hp_type": "float", "hp_name": "learning_rate", "hp_value": [1e-2, 1]},
            {"hp_type": "int", "hp_name": "depth", "hp_value": [3, 10]},
            {"hp_type": "float", "hp_name": "l2_leaf_reg", "hp_value": [1, 5]},
            {
                "hp_type": "cat",
                "hp_name": "loss_function",
                "hp_value": ["Logloss", "CrossEntropy"],
            },
            {
                "hp_type": "cat",
                "hp_name": "feature_border_type",
                "hp_value": ["GreedyLogSum", "Median", "MinEntropy"],
            },
            {
                "hp_type": "int",
                "hp_name": "fold_permutation_block",
                "hp_value": [1, 10],
            },
            {"hp_type": "cat", "hp_name": "logging_level", "hp_value": "Silent"},
            {
                "hp_type": "float",
                "hp_name": "bagging_temperature",
                "hp_value": [1e-5, 1],
            },
            {"hp_type": "cat", "hp_name": "allow_writing_files", "hp_value": False},
        ]
    elif model_name == "ImbalancedRandomForestClassifier":
        estimator = BalancedRandomForestClassifier
        parameters = [
            {"hp_type": "int", "hp_name": "n_estimators", "hp_value": [40, 120]},
            {"hp_type": "cat", "hp_name": "criterion", "hp_value": ["gini", "entropy"]},
            {"hp_type": "int", "hp_name": "max_depth", "hp_value": [5, 100]},
            {"hp_type": "int", "hp_name": "min_samples_split", "hp_value": [2, 50]},
            {"hp_type": "int", "hp_name": "min_samples_leaf", "hp_value": [1, 10]},
            {
                "hp_type": "float",
                "hp_name": "min_weight_fraction_leaf",
                "hp_value": [1e-5, 0.5],
            },
            {"hp_type": "cat", "hp_name": "max_features", "hp_value": ["sqrt", "log2"]},
            {
                "hp_type": "cat",
                "hp_name": "class_weight",
                "hp_value": [None, "balanced"],
            },
            {
                "hp_type": "cat",
                "hp_name": "sampling_strategy",
                "hp_value": ["majority", "all", "not majority"],
            },
        ]
    else:
        raise ValueError(
            f"Provided {model_name} is not available, please use one of the following {', '.join(_AVAILABLE_MODELS)}."
        )
    return estimator, parameters
