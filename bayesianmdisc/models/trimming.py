import torch
from torch import vmap

from bayesianmdisc.models import ModelProtocol

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.statistics.metrics import (
    coefficient_of_determination,
    root_mean_squared_error,
)
from bayesianmdisc.data import (
    DeformationInputs,
    StressOutputs,
    TestCases,
)
from bayesianmdisc.errors import ModelTrimmingError

file_name_parameters = "relevant_parameters.txt"


def trim_model(
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
    parameter_names = model.get_parameter_names()

    if metric == "rmse":
        metric_name = metric
        metric_func = root_mean_squared_error
    elif metric == "R2":
        metric_name = metric
        metric_func = coefficient_of_determination
    else:
        raise ModelTrimmingError(
            f"No implementation for requested error metric: {metric}"
        )

    def order_parameters_according_to_relevance() -> list[int]:
        mean_parameters = torch.mean(parameter_samples, dim=0)
        _, indices = torch.sort(mean_parameters)
        return indices.tolist()

    def evaluate_mean_model_ouputs() -> Tensor:
        vmap_func = lambda parameters: model(inputs, test_cases, parameters)
        predictions = vmap(vmap_func)(parameter_samples)
        return torch.mean(predictions, dim=0)

    def determine_model_accuracy() -> float:
        mean_model_outputs = evaluate_mean_model_ouputs()
        return metric_func(mean_model_outputs, outputs)

    def calculate_absolute_relative_deterioration(
        accuracy: float, trimmed_accuracy: float
    ):
        return abs(accuracy - trimmed_accuracy) / accuracy

    def print_initial_info(accuracy: float) -> None:
        accuracy = round(accuracy, 6)
        print("Start trimming model ...")
        print(f"Initial accuracy: {metric_name}={accuracy}")

    def trim_condition(deterioration: float) -> bool:
        return deterioration < relative_thresshold

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

    parameter_indices = order_parameters_according_to_relevance()

    accuracy = determine_model_accuracy()
    print_initial_info(accuracy)
    deterioration = 0.0
    for parameter_index in parameter_indices:
        parameter_name = parameter_names[parameter_index]
        model.deactivate_parameters([parameter_index])
        trimmed_accuracy = determine_model_accuracy()
        deterioration = calculate_absolute_relative_deterioration(
            accuracy, trimmed_accuracy
        )
        if trim_condition(deterioration):
            print_info(parameter_name, accuracy, deterioration)
            accuracy = trimmed_accuracy
        else:
            model.activate_parameters([parameter_index])

    save_relevant_parameter_names()
    print_relevant_parameter_names()
