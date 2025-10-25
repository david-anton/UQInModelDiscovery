from typing import TypeAlias

import numpy as np
from sklearn.metrics import mean_absolute_error as mean_absolute_error_sklearn
from sklearn.metrics import r2_score as r2_score_sklearn
from sklearn.metrics import root_mean_squared_error as root_mean_squared_error_sklean

from uqmodeldisc.customtypes import NPArray, Tensor
from uqmodeldisc.statistics.utility import determine_quantiles_from_samples

Values: TypeAlias = Tensor | NPArray


def convert_to_numpy(tensor: Tensor) -> NPArray:
    return tensor.detach().cpu().numpy()


def convert_to_numpy_if_necessary(values: Values) -> NPArray:
    if isinstance(values, Tensor):
        return convert_to_numpy(values)
    else:
        return values


def flatten(array: NPArray) -> NPArray:
    return array.flatten()


def coefficient_of_determination(y_predicted: Values, y_true: Values) -> float:
    y_predicted_ = flatten(convert_to_numpy_if_necessary(y_predicted))
    y_true_ = flatten(convert_to_numpy_if_necessary(y_true))
    return r2_score_sklearn(y_true=y_true_, y_pred=y_predicted_)


def root_mean_squared_error(y_predicted: Values, y_true: Values) -> float:
    y_predicted_ = convert_to_numpy_if_necessary(y_predicted)
    y_true_ = convert_to_numpy_if_necessary(y_true)
    return root_mean_squared_error_sklean(y_true=y_true_, y_pred=y_predicted_)


def mean_absolute_error(y_predicted: Values, y_true: Values) -> float:
    y_predicted_ = convert_to_numpy_if_necessary(y_predicted)
    y_true_ = convert_to_numpy_if_necessary(y_true)
    return mean_absolute_error_sklearn(y_true=y_true_, y_pred=y_predicted_)


def model_coverage_test(
    y_predicted_samples: Values, y_true: Values, credible_interval: float
) -> float:

    def _test_coverage(samples: NPArray, true_values: NPArray) -> NPArray:
        min_quantiles, max_quantiles = determine_quantiles_from_samples(
            samples, credible_interval
        )
        is_larger_or_equal_min = true_values >= min_quantiles
        is_smaller_or_equal_max = true_values <= max_quantiles
        return np.logical_and(is_larger_or_equal_min, is_smaller_or_equal_max)

    y_predicted_samples_ = convert_to_numpy_if_necessary(y_predicted_samples)
    y_true_ = convert_to_numpy_if_necessary(y_true)

    coverage_test_results = _test_coverage(y_predicted_samples_, y_true_)
    return _evaluate_coverage_test_results(coverage_test_results)


def gp_coverage_test(
    y_min_quantiles: Values, y_max_quantiles: Values, y_true: Values
) -> float:

    def _test_coverage(
        min_quantiles: NPArray, max_quantiles: NPArray, true_values: NPArray
    ) -> NPArray:
        is_larger_or_equal_min = true_values >= min_quantiles
        is_smaller_or_equal_max = true_values <= max_quantiles
        return np.logical_and(is_larger_or_equal_min, is_smaller_or_equal_max)

    y_min_quantiles_ = convert_to_numpy_if_necessary(y_min_quantiles)
    y_max_quantiles_ = convert_to_numpy_if_necessary(y_max_quantiles)
    y_true_ = convert_to_numpy_if_necessary(y_true)

    coverage_test_results = _test_coverage(y_min_quantiles_, y_max_quantiles_, y_true_)
    return _evaluate_coverage_test_results(coverage_test_results)


def _evaluate_coverage_test_results(coverage_test_results: NPArray) -> float:
    return 100.0 * float(np.mean(coverage_test_results))
