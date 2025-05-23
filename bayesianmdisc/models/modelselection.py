from typing import TypeAlias, Any

import torch
from torch import vmap
import numpy as np

from bayesianmdisc.customtypes import Tensor, NPArray
from bayesianmdisc.data import (
    DeformationInputs,
    StressOutputs,
    TestCases,
)
from SALib import ProblemSpec
from bayesianmdisc.errors import ModelTrimmingError
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import ModelProtocol
from bayesianmdisc.models.base import ParameterIndex, ParameterIndices
from bayesianmdisc.statistics.metrics import (
    coefficient_of_determination,
    mean_absolute_error,
    root_mean_squared_error,
)
from bayesianmdisc.bayes.distributions import DistributionProtocol

ModelAccuracies: TypeAlias = list[float]


file_name_parameters = "relevant_parameters.txt"


def select_model_through_backward_elimination(
    model: ModelProtocol,
    metric: str,
    relative_thresshold: float,
    parameter_samples: Tensor,
    inputs: DeformationInputs,
    test_cases: TestCases,
    outputs: StressOutputs,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    model.reset_parameter_deactivations()
    parameter_names = model.parameter_names

    if metric == "mae":
        metric_name = metric
        metric_func = mean_absolute_error
        sort_metric_descending = False
    elif metric == "rmse":
        metric_name = metric
        metric_func = root_mean_squared_error
        sort_metric_descending = False
    elif metric == "R2":
        metric_name = metric
        metric_func = coefficient_of_determination
        sort_metric_descending = True
    else:
        raise ModelTrimmingError(
            f"No implementation for requested error metric: {metric}"
        )

    def find_index_of_most_irrelevant_parameter() -> ParameterIndex:
        active_parameter_indices = model.get_active_parameter_indices()
        accuracies = run_leave_one_out_cross_validation(active_parameter_indices)
        return filter_most_irrelevant_active_parameter_index(
            active_parameter_indices, accuracies
        )

    def run_leave_one_out_cross_validation(
        active_parameter_indices: ParameterIndices,
    ) -> ModelAccuracies:
        model_accuracies = []
        for parameter_indice in active_parameter_indices:
            model.deactivate_parameters([parameter_indice])
            model_accuracy = determine_model_accuracy()
            model_accuracies += [model_accuracy]
            model.activate_parameters([parameter_indice])
        return model_accuracies

    def determine_model_accuracy() -> float:
        mean_model_outputs = evaluate_mean_model_ouputs()
        return metric_func(mean_model_outputs, outputs)

    def evaluate_mean_model_ouputs() -> Tensor:
        vmap_func = lambda parameters: model(inputs, test_cases, parameters)
        predictions = vmap(vmap_func)(parameter_samples)
        return torch.mean(predictions, dim=0)

    def filter_most_irrelevant_active_parameter_index(
        active_parameter_indices: ParameterIndices,
        accuracies: ModelAccuracies,
    ) -> ParameterIndex:
        indices_of_sorted_accuracies = torch.sort(
            torch.tensor(accuracies), descending=sort_metric_descending
        )[1].tolist()
        index_of_least_deterioration = indices_of_sorted_accuracies[0]
        return active_parameter_indices[index_of_least_deterioration]

    def print_initial_info(accuracy: float) -> None:
        accuracy = round(accuracy, 6)
        print("Start trimming model ...")
        print(f"Initial accuracy: {metric_name}={accuracy}")

    def calculate_absolute_relative_deterioration(
        initial_accuracy: float, accuracy: float
    ):
        return abs(initial_accuracy - accuracy) / initial_accuracy

    def trim_condition(deterioration: float) -> bool:
        return deterioration < relative_thresshold

    def print_info(parameter_name: str, accuracy: float, deterioration: float) -> None:
        accuracy = round(accuracy, 6)
        deterioration = round(deterioration, 6)
        print(f"Trimm parameter {parameter_name}")
        print(
            f"Remaining accuracy: {metric_name}={accuracy}, deterioration: {deterioration}"
        )

    initial_accuracy = determine_model_accuracy()
    print_initial_info(initial_accuracy)
    deterioration = 0.0
    while trim_condition(deterioration):
        parameter_index = find_index_of_most_irrelevant_parameter()
        parameter_name = parameter_names[parameter_index]
        model.deactivate_parameters([parameter_index])
        accuracy = determine_model_accuracy()
        deterioration = calculate_absolute_relative_deterioration(
            initial_accuracy, accuracy
        )
        if trim_condition(deterioration):
            print_info(parameter_name, accuracy, deterioration)
        else:
            model.activate_parameters([parameter_index])

    save_relevant_parameter_names(model, output_subdirectory, project_directory)
    print_relevant_parameter_names(model)


def select_model_through_sensitivity_analysis(
    model: ModelProtocol,
    parameter_distribution: DistributionProtocol,
    thresshold: float,
    num_samples: int,
    inputs: DeformationInputs,
    test_cases: TestCases,
    outputs: StressOutputs,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    num_parameters = model.num_parameters
    parameter_names = model.parameter_names
    num_inputs = len(inputs)
    output_dim = outputs.shape[1]
    calc_second_order = False
    num_samples_bounds = 8192

    # Correct the number of samples as required to calculate Sobol indices
    # num_samples = num_samples * (2 * num_parameters)

    parameter_samples = parameter_distribution.sample(num_samples_bounds)

    def print_initial_info() -> None:
        print("Start sensitivity analysis ...")

    def determine_parameter_bounds(parameter_samples: NPArray) -> Any:
        parameter_samples = parameter_samples.transpose((1, 0))
        lower_bound = np.amin(parameter_samples, axis=1, keepdims=True)
        upper_bound = np.amax(parameter_samples, axis=1, keepdims=True)
        return np.hstack((lower_bound, upper_bound)).tolist()

    def evaluate_model(parameter_samples: Tensor) -> StressOutputs:
        vmap_model_func = lambda _parameter_sample: model(
            inputs, test_cases, _parameter_sample
        )
        return vmap(vmap_model_func)(parameter_samples)

    def reshape_model_outputs(model_outputs: StressOutputs) -> StressOutputs:
        model_outputs = torch.transpose(model_outputs, 0, 1)
        model_outputs = torch.transpose(model_outputs, 1, 2)
        return model_outputs

    print_initial_info()

    parameter_samples_spec = parameter_samples.detach().cpu().numpy()
    parameter_bounds_spec = determine_parameter_bounds(parameter_samples_spec)

    problem = ProblemSpec(
        {
            "num_vars": num_parameters,
            "names": parameter_names,
            "bounds": parameter_bounds_spec,
            "outputs": ["P1"],
        }
    )
    problem.sample_saltelli(num_samples, calc_second_order=calc_second_order)
    parameter_samples = torch.from_numpy(problem.samples)

    model_outputs = evaluate_model(parameter_samples)
    model_outputs = reshape_model_outputs(model_outputs)
    one_model_output = model_outputs[14, 0, :].detach().cpu().numpy()

    # problem.set_samples(parameter_samples_spec)
    problem.set_results(one_model_output)
    sobol_indices = problem.analyze_sobol(calc_second_order=calc_second_order)

    print(sobol_indices.analysis["S1"])
    print(np.sum(sobol_indices.analysis["S1"]))
    print(sobol_indices.analysis["ST"])
    print(np.sum(sobol_indices.analysis["ST"]))

    save_relevant_parameter_names(model, output_subdirectory, project_directory)
    print_relevant_parameter_names(model)


def save_relevant_parameter_names(
    model: ModelProtocol,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    relevant_parameter_names = model.get_active_parameter_names()
    output_path = project_directory.create_output_file_path(
        file_name_parameters, output_subdirectory
    )

    with open(output_path, mode="w") as output_file:
        output_file.write("\n".join(relevant_parameter_names) + "\n")


def print_relevant_parameter_names(model: ModelProtocol) -> None:
    relevant_parameter_names = model.get_active_parameter_names()

    for parameter_name in relevant_parameter_names:
        print(parameter_name)
