from typing import TypeAlias
import math

from sklearn.metrics import (
    r2_score,
)
import numpy as np
from sklearn.metrics import root_mean_squared_error as root_mean_squared_error_sklean

from bayesianmdisc.customtypes import NPArray, Tensor
from bayesianmdisc.errors import StatisticalMetricError

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
    return r2_score(y_true=y_true_, y_pred=y_predicted_)


def root_mean_squared_error(y_predicted: Values, y_true: Values) -> float:
    y_predicted_ = convert_to_numpy_if_necessary(y_predicted)
    y_true_ = convert_to_numpy_if_necessary(y_true)
    return root_mean_squared_error_sklean(y_true=y_true_, y_pred=y_predicted_)


def coverage_test(
    y_predicted_samples: Values, y_true: Values, credible_interval: float
) -> float:

    def _validate_credible_interval(credible_interval: float) -> None:
        is_larger_or_equal_zero = credible_interval >= 0.0
        is_smaller_or_equal_one = credible_interval <= 1.0
        is_valid = is_larger_or_equal_zero and is_smaller_or_equal_one

        if not is_valid:
            raise StatisticalMetricError(
                f"""The credible interval is expected to be positive 
                and smaller or equal than one, but is {credible_interval}"""
            )

    def _test_coverage(samples: NPArray, true_value: float) -> bool:

        def _sort_samples() -> NPArray:
            return np.sort(samples)

        def _determine_indices() -> tuple[int, int]:
            num_samples = len(samples)
            max_index = num_samples - 1
            quantile = (1.0 - credible_interval) / 2
            lower_index = int(math.ceil(quantile * max_index))
            upper_index = int(math.floor((1.0 - quantile) * max_index))
            return lower_index, upper_index

        def _test(sorted_samples: NPArray, lower_index: int, upper_index: int) -> bool:
            min_value = sorted_samples[lower_index]
            max_value = sorted_samples[upper_index]
            is_larger_or_equal_min = true_value >= min_value
            is_smaller_or_equal_max = true_value <= max_value
            is_covered = is_larger_or_equal_min and is_smaller_or_equal_max
            return is_covered

        sorted_samples = _sort_samples()
        lower_index, upper_index = _determine_indices()
        return _test(sorted_samples, lower_index, upper_index)

    def _evaluate_coverage_test_results(coverage_test_results: list[bool]) -> float:
        coverage_test_results_np = np.array(coverage_test_results)
        return 100.0 * float(np.mean(coverage_test_results_np))

    _validate_credible_interval(credible_interval)
    y_predicted_samples_ = list(convert_to_numpy_if_necessary(y_predicted_samples))
    y_true_ = list(convert_to_numpy_if_necessary(y_true))

    coverage_test_results = []
    for samples, true_value in zip(y_predicted_samples_, y_true_):
        coverage_test_result = _test_coverage(samples, true_value)
        coverage_test_results += [coverage_test_result]

    return _evaluate_coverage_test_results(coverage_test_results)
