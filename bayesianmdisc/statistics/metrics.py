from typing import TypeAlias

from sklearn.metrics import (
    r2_score,
)
from sklearn.metrics import root_mean_squared_error as root_mean_squared_error_sklean

from bayesianmdisc.types import NPArray, Tensor

Values: TypeAlias = Tensor | NPArray


def convert_to_numpy(tensor: Tensor) -> NPArray:
    return tensor.detach().cpu().numpy()


def convert_to_numpy_if_necessary(values: Values) -> NPArray:
    if isinstance(values, Tensor):
        return convert_to_numpy(values)
    else:
        return values


def coefficient_of_determination(y_predicted: Values, y_true: Values) -> float:
    y_predicted_ = convert_to_numpy_if_necessary(y_predicted)
    y_true_ = convert_to_numpy_if_necessary(y_true)
    return r2_score(y_true=y_true_, y_pred=y_predicted_)


def root_mean_squared_error(y_predicted: Values, y_true: Values) -> float:
    y_predicted_ = convert_to_numpy_if_necessary(y_predicted)
    y_true_ = convert_to_numpy_if_necessary(y_true)
    return root_mean_squared_error_sklean(y_true=y_true_, y_pred=y_predicted_)
