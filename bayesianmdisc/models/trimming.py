import torch
from torch import vmap

from bayesianmdisc.models import ModelProtocol

from bayesianmdisc.customtypes import Tensor
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.statistics.metrics import coefficient_of_determination
from bayesianmdisc.data import (
    DeformationInputs,
    StressOutputs,
    TestCases,
)

file_name_parameters = "relevant_parameters.txt"


def trim_model(
    model: ModelProtocol,
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
    metric = "R2"

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
        return coefficient_of_determination(mean_model_outputs, outputs)

    def calculate_relative_deterioration(accuracy: float, trimmed_accuracy: float):
        return (accuracy - trimmed_accuracy) / accuracy

    def print_initial_info(accuracy: float) -> None:
        accuracy = round(accuracy, 4)
        print("Start trimming model ...")
        print(f"Initial accuracy: {metric}={accuracy}")

    def print_info(parameter_name: str, accuracy: float, deterioration: float) -> None:
        accuracy = round(accuracy, 4)
        deterioration = round(deterioration, 4)
        print(f"Trimm parameter {parameter_name}")
        print(
            f"Remaining accuracy: {metric}={accuracy}, deterioration: {deterioration}"
        )

    def save_relevant_parameter_names() -> None:
        relevant_parameter_names = model.get_active_parameter_names()
        output_path = project_directory.create_output_file_path(
            file_name_parameters, output_subdirectory
        )

        with open(output_path, mode="w") as output_file:
            output_file.write("\n".join(relevant_parameter_names) + "\n")

    parameter_indices = order_parameters_according_to_relevance()

    accuracy = determine_model_accuracy()
    print_initial_info(accuracy)
    deterioration = 0.0
    parameter = 0
    while deterioration < relative_thresshold:
        parameter_index = parameter_indices[parameter]
        parameter_name = parameter_names[parameter_index]
        model.deactivate_parameters([parameter_index])
        trimmed_accuracy = determine_model_accuracy()
        deterioration = calculate_relative_deterioration(accuracy, trimmed_accuracy)
        accuracy = trimmed_accuracy
        parameter += 1
        print_info(parameter_name, accuracy, deterioration)

    save_relevant_parameter_names()
