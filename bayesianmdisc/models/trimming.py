from typing import TypeAlias

import torch
from torch import vmap

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.data import (
    DeformationInputs,
    StressOutputs,
    TestCases,
)
from bayesianmdisc.errors import ModelTrimmingError
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import ModelProtocol
from bayesianmdisc.models.base import ParameterIndices, ParameterIndex
from bayesianmdisc.statistics.metrics import (
    coefficient_of_determination,
    root_mean_squared_error,
)

ModelAccuracies: TypeAlias = list[float]


file_name_parameters = "relevant_parameters.txt"


def trim_model(
    model: ModelProtocol,
    metric: str,
    relative_deterioration_thresshold: float,
    parameter_samples: Tensor,
    inputs: DeformationInputs,
    test_cases: TestCases,
    outputs: StressOutputs,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    model.reset_parameter_deactivations()
    parameter_names = model.parameter_names

    if metric == "rmse":
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
        return deterioration < relative_deterioration_thresshold

    def print_info(parameter_name: str, accuracy: float, deterioration: float) -> None:
        accuracy = round(accuracy, 6)
        deterioration = round(deterioration, 6)
        print(f"Trimm parameter {parameter_name}")
        print(
            f"Remaining accuracy: {metric_name}={accuracy}, deterioration: {deterioration}"
        )

    def save_relevant_parameter_names() -> None:
        relevant_parameter_names = model.get_active_parameter_names()
        output_path = project_directory.create_output_file_path(
            file_name_parameters, output_subdirectory
        )

        with open(output_path, mode="w") as output_file:
            output_file.write("\n".join(relevant_parameter_names) + "\n")

    def print_relevant_parameter_names() -> None:
        relevant_parameter_names = model.get_active_parameter_names()

        for parameter_name in relevant_parameter_names:
            print(parameter_name)

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

    save_relevant_parameter_names()
    print_relevant_parameter_names()
