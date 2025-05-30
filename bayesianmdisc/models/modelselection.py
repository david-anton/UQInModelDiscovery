from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from SALib import ProblemSpec
from torch import vmap

from bayesianmdisc.bayes.distributions import DistributionProtocol
from bayesianmdisc.customtypes import Device, NPArray, Tensor
from bayesianmdisc.data import (
    DeformationInputs,
    StressOutputs,
    TestCases,
    data_set_label_linka,
    data_set_label_treloar,
    zero_stress_inputs_linka,
    zero_stress_inputs_treloar,
)
from bayesianmdisc.data.testcases import (
    map_test_case_identifiers_to_labels,
    TestCaseLabels,
)
from bayesianmdisc.errors import ModelSelectionError
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.readerswriters import PandasDataWriter
from bayesianmdisc.models.base import ParameterIndex, ParameterIndices, ModelProtocol
from bayesianmdisc.statistics.metrics import (
    coefficient_of_determination,
    mean_absolute_error,
    root_mean_squared_error,
)

ModelAccuracies: TypeAlias = list[float]
Problem: TypeAlias = ProblemSpec
SIndices: TypeAlias = NPArray
SIndicesList: TypeAlias = list[SIndices]

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
        raise ModelSelectionError(
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


first_indices_label = "first_sobol_indices"
total_indices_label = "total_sobol_indices"
pd_column_lable_test_cases = "test cases"


def select_model_through_sobol_sensitivity_analysis(
    model: ModelProtocol,
    parameter_distribution: DistributionProtocol,
    first_sobol_index_thresshold: float,
    num_samples_factor: int,
    data_set_label: str,
    inputs: DeformationInputs,
    test_cases: TestCases,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    calc_second_order = False
    num_samples_bounds = 8192
    num_parameters = model.num_parameters
    parameter_names = model.parameter_names
    num_outputs = model.output_dim

    data_writer = PandasDataWriter(project_directory)

    if data_set_label == data_set_label_treloar:
        skipped_input_indices = zero_stress_inputs_treloar
    elif data_set_label == data_set_label_linka:
        skipped_input_indices = zero_stress_inputs_linka
    else:
        raise ModelSelectionError(
            """There is no implementation for the specified data set."""
        )

    def print_initial_info() -> None:
        print("Start sensitivity analysis ...")

    def remove_skipped_inputs(
        inputs: DeformationInputs, test_cases: TestCases
    ) -> tuple[DeformationInputs, TestCases]:
        _inputs = inputs
        _test_cases = test_cases
        for input_index in sorted(skipped_input_indices, reverse=True):
            _inputs = torch.concat(
                (_inputs[:input_index, :], _inputs[input_index + 1 :, :])
            )
            _test_cases = torch.concat(
                (_test_cases[:input_index], _test_cases[input_index + 1 :])
            )
        return _inputs, _test_cases

    def determine_parameter_bounds() -> NPArray:
        parameter_samples_torch = parameter_distribution.sample(num_samples_bounds)
        parameter_samples = from_torch_to_numpy(parameter_samples_torch)
        parameter_samples = parameter_samples.transpose((1, 0))
        lower_bound = np.amin(parameter_samples, axis=1, keepdims=True)
        upper_bound = np.amax(parameter_samples, axis=1, keepdims=True)
        return np.hstack((lower_bound, upper_bound))

    print_initial_info()
    inputs, test_cases = remove_skipped_inputs(inputs, test_cases)
    num_inputs = len(inputs)
    test_case_labels = map_test_case_identifiers_to_labels(test_cases)
    parameter_bounds = determine_parameter_bounds().tolist()

    first_indices_outputs_list: SIndicesList = []
    total_indices_outputs_list: SIndicesList = []

    for output_index in range(num_outputs):

        def define_problem() -> Problem:
            output_name = f"output_{output_index}"
            return ProblemSpec(
                {
                    "num_vars": num_parameters,
                    "names": parameter_names,
                    "bounds": parameter_bounds,
                    "outputs": [output_name],
                }
            )

        def sample(problem: Problem) -> NPArray:
            problem.sample_saltelli(
                num_samples_factor, calc_second_order=calc_second_order
            )
            return problem.samples

        def evaluate_model(parameter_samples: NPArray) -> NPArray:
            parameter_samples_torch = from_numpy_to_torch(parameter_samples, device)
            vmap_model_func = lambda parameter_sample: model(
                inputs, test_cases, parameter_sample
            )
            outputs_torch = vmap(vmap_model_func)(parameter_samples_torch)
            return from_torch_to_numpy(outputs_torch)

        def reshape_model_outputs(outputs: NPArray) -> NPArray:
            return np.transpose(outputs, (1, 2, 0))

        problem = define_problem()
        parameter_samples = sample(problem)
        model_outputs = evaluate_model(parameter_samples)
        model_outputs = reshape_model_outputs(model_outputs)

        first_indices_inputs_list: SIndicesList = []
        total_indices_inputs_list: SIndicesList = []

        for input_index in range(num_inputs):

            def print_info() -> None:
                print(
                    f"Analyze sensitivities of output {output_index} at input {input_index} ..."
                )

            def analyze_sobol_indices(problem: Problem) -> None:
                model_output = model_outputs[input_index, output_index, :]
                problem.set_results(model_output)
                problem.analyze_sobol(calc_second_order=calc_second_order)

            def get_indices(problem: Problem) -> tuple[SIndices, SIndices]:
                first_indices = problem.analysis["S1"]
                total_indices = problem.analysis["ST"]
                return first_indices, total_indices

            def remove_independent_inputs(
                indices: SIndices, test_cases_labels: TestCaseLabels
            ) -> tuple[SIndices, TestCaseLabels]:
                cumulated_indices = np.sum(indices, axis=1)
                is_cumulated_indice_nan = np.isnan(cumulated_indices)
                is_cumulated_indice_zero = np.isclose(cumulated_indices, 0.0, rtol=1e-5)

                is_independent_input = np.logical_or(
                    is_cumulated_indice_nan, is_cumulated_indice_zero
                )
                is_dependent_input = np.invert(is_independent_input)

                dependent_input_indices = (
                    np.where(is_dependent_input)[0].reshape((-1,)).tolist()
                )
                relevant_indices = indices[dependent_input_indices, :]
                relevant_test_case_labels = [
                    test_cases_labels[index] for index in dependent_input_indices
                ]
                return relevant_indices, relevant_test_case_labels

            def calculate_statistics(indices: SIndices) -> tuple[NPArray, NPArray]:
                means = np.mean(indices, axis=0)
                stdes = np.std(indices, axis=0) / np.sqrt(num_inputs)
                return means, stdes

            print_info()
            analyze_sobol_indices(problem)
            first_indices, total_indices = get_indices(problem)
            first_indices_inputs_list += [first_indices]
            total_indices_inputs_list += [total_indices]

        first_indices_inputs = np.vstack(first_indices_inputs_list)
        total_indices_inputs = np.vstack(total_indices_inputs_list)

        relevant_first_indices_inputs, relevant_first_indices_test_case_labels = (
            remove_independent_inputs(first_indices_inputs, test_case_labels)
        )
        relevant_total_indices_inputs, relevant_total_indices_test_case_labels = (
            remove_independent_inputs(total_indices_inputs, test_case_labels)
        )

        mean_first_indices_inputs, stde_first_indices_inputs = calculate_statistics(
            relevant_first_indices_inputs
        )
        mean_total_indices_inputs, stde_total_indices_inputs = calculate_statistics(
            relevant_total_indices_inputs
        )

        def save_analysis_results(
            indices_inputs: SIndices,
            test_case_labels: TestCaseLabels,
            indices_label: str,
        ) -> None:
            file_name = join_analysis_results_file_name(indices_label, output_index)
            data_frame = pd.DataFrame(indices_inputs, columns=parameter_names)
            data_frame.insert(0, pd_column_lable_test_cases, test_case_labels)
            data_writer.write(
                data_frame,
                file_name,
                output_subdirectory,
                header=True,
            )

        def save_analysis_results_statistics(
            mean_indices_inputs: NPArray,
            stde_indices_inputs: NPArray,
            indices_label: str,
        ) -> None:
            statistics = np.vstack((mean_indices_inputs, stde_indices_inputs))
            statistics_lables = ["mean", "standard error"]
            file_name = f"{indices_label}_statistics_output_{output_index}"
            data_frame = pd.DataFrame(statistics, columns=parameter_names)
            data_frame.insert(0, "", statistics_lables)
            data_writer.write(
                data_frame,
                file_name,
                output_subdirectory,
                header=True,
            )

        save_analysis_results(
            relevant_first_indices_inputs,
            relevant_first_indices_test_case_labels,
            first_indices_label,
        )
        save_analysis_results(
            relevant_total_indices_inputs,
            relevant_total_indices_test_case_labels,
            total_indices_label,
        )
        save_analysis_results_statistics(
            mean_first_indices_inputs,
            stde_first_indices_inputs,
            first_indices_label,
        )
        save_analysis_results_statistics(
            mean_total_indices_inputs,
            stde_total_indices_inputs,
            total_indices_label,
        )
        first_indices_outputs_list += [mean_first_indices_inputs]
        total_indices_outputs_list += [mean_total_indices_inputs]

    def select_relevant_parameter_indices(
        mean_first_indices_outputs: SIndices,
    ) -> ParameterIndices:
        return (
            np.where(mean_first_indices_outputs >= first_sobol_index_thresshold)[0]
            .reshape((-1,))
            .tolist()
        )

    def deactivate_irrelevant_parameters(
        relevant_parameter_indices: ParameterIndices,
    ) -> None:
        model.deactivate_parameters([_ for _ in range(num_parameters)])
        model.activate_parameters(relevant_parameter_indices)
        save_relevant_parameter_names(model, output_subdirectory, project_directory)
        print_relevant_parameter_names(model)

    first_indices_outputs = np.vstack(first_indices_outputs_list)
    total_indices_outputs = np.vstack(total_indices_outputs_list)
    mean_first_indices_outputs = np.mean(first_indices_outputs, axis=0)
    _ = np.mean(total_indices_outputs, axis=0)

    relevant_parameter_indices = select_relevant_parameter_indices(
        mean_first_indices_outputs
    )
    deactivate_irrelevant_parameters(relevant_parameter_indices)


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


def from_numpy_to_torch(array: NPArray, device: Device) -> Tensor:
    return torch.from_numpy(array).type(torch.get_default_dtype()).to(device)


def from_torch_to_numpy(tensor: Tensor) -> NPArray:
    return tensor.detach().cpu().numpy()


def join_analysis_results_file_name(file_name_prefix: str, output_index: int) -> str:
    return f"{file_name_prefix}_output_{output_index}"
