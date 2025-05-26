from typing import Any, Dict, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import torch

from bayesianmdisc.customtypes import NPArray, PDDataFrame
from bayesianmdisc.data.testcases import (
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
    map_test_case_identifiers_to_labels,
)
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.readerswriters import CSVDataReader
from bayesianmdisc.postprocessing.plot.utility import split_treloar_inputs_and_outputs

ParameterIndices: TypeAlias = list[int]

first_indices_label = "first_sobol_indices"
total_indices_label = "total_sobol_indices"
pd_column_lable_test_cases = "test cases"


class IndicesDevelopmentPlotterConfigTreloar:
    def __init__(self) -> None:
        # label size
        self.label_size = 10
        # font size in legend
        self.font_size = 12
        self.font: Dict[str, Any] = {"size": self.font_size}

        # major ticks
        self.major_tick_label_size = 12
        self.major_ticks_size = self.font_size
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 12
        self.minor_ticks_size = 12
        self.minor_ticks_width = 1

        # titles
        self.title_ut = "uniaxial tension"
        self.title_ebt = "equibiaxial tension"
        self.title_ps = "pure shear"

        # labels
        self.xaxis_label = "stretch [-]"
        self.yaxis_label_first_indices = "first Sobol indice [-]"
        self.yaxis_label_total_indices = "total Sobol indice [-]"

        # results
        self.color_map = "viridis"
        self.marker = "o"
        self.marker_size = 5

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_sobol_indices_results_treloar(
    relevant_parameter_indices: ParameterIndices,
    indices_label: str,
    input_file_name: str,
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    pass
    num_relevant_parameters = len(relevant_parameter_indices)
    parameter_names, indices_results = read_indices_results(
        relevant_parameter_indices=relevant_parameter_indices,
        input_file_name=input_file_name,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
    )

    input_sets, test_case_identifiers, _ = split_treloar_inputs_and_outputs(
        inputs, test_cases, outputs
    )
    input_sets = remove_first_inputs(input_sets)

    def plot_one_indices_development(
        inputs: NPArray, test_case_identifier: int
    ) -> None:
        config = IndicesDevelopmentPlotterConfigTreloar()
        file_name = f"treloar_data_{indices_label}.png"
        results = filter_indices_results_for_test_case(
            indices_results, test_case_identifier, parameter_names
        )

        figure, axes = plt.subplots()
        color_map = plt.get_cmap(config.color_map, num_relevant_parameters)
        colors = color_map(np.linspace(0, 1, num_relevant_parameters))

        data_stretches = inputs[:, 0]
        min_stretch = np.amin(data_stretches)
        max_stretch = np.amax(data_stretches)

        for parameter_index in range(num_relevant_parameters):
            color = colors[parameter_index]
            parameter_name_plot = parameter_names[parameter_index]
            results_plot = results[:, parameter_index]
            axes.plot(
                data_stretches,
                results_plot,
                marker=config.marker,
                color=color,
                markersize=config.marker_size,
                linestyle="dashdot",
                label=parameter_name_plot,
            )

        # axis ticks
        x_ticks = np.linspace(min_stretch, max_stretch, num=6)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_tick_labels)

        # axis labels
        axes.set_xlabel(config.xaxis_label, **config.font)
        if indices_label == first_indices_label:
            y_label = config.yaxis_label_first_indices
        elif indices_label == total_indices_label:
            y_label = config.yaxis_label_total_indices
        axes.set_ylabel(y_label, **config.font)

        # legend
        axes.legend(
            fontsize=config.font_size,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # title
        if test_case_identifier == test_case_identifier_uniaxial_tension:
            title = config.title_ut
        elif test_case_identifier == test_case_identifier_equibiaxial_tension:
            title = config.title_ebt
        elif test_case_identifier == test_case_identifier_pure_shear:
            title = config.title_ps
        axes.set_title(title, **config.font)

        # saving
        output_path = project_directory.create_output_file_path(
            file_name=file_name, subdir_name=output_subdirectory
        )
        figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)

    for input_set, test_case_identifier in zip(input_sets, test_case_identifiers):
        plot_one_indices_development(input_set, test_case_identifier)


def read_indices_results(
    relevant_parameter_indices: ParameterIndices,
    input_file_name: str,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> tuple[list[str], PDDataFrame]:
    data_reader = CSVDataReader(project_directory)

    def filter_relevant_parameter_names(data_frame: PDDataFrame) -> list[str]:
        all_parameter_names = data_frame.columns[1:].tolist()
        return all_parameter_names[relevant_parameter_indices]

    def filter_relevant_results(
        data_frame: PDDataFrame, parameter_names: list[str]
    ) -> PDDataFrame:
        return data_frame[parameter_names]

    data_frame = data_reader.read_as_pandas_data_frame(
        input_file_name, output_subdirectory, read_from_output_dir=True
    )
    parameter_names = filter_relevant_parameter_names(data_frame)
    results = filter_relevant_results(data_frame, parameter_names)
    return parameter_names, results


def remove_first_inputs(input_sets: list[NPArray]) -> list[NPArray]:
    return [input_set[1:-1, :] for input_set in input_sets]


def filter_indices_results_for_test_case(
    indices_results: PDDataFrame, test_case_identifier: int, parameter_names: list[str]
) -> NPArray:
    test_case_label = map_test_case_identifiers_to_labels(
        torch.tensor([test_case_identifier])
    )[0]
    filtered_indices_results = indices_results[
        indices_results[pd_column_lable_test_cases] == test_case_label
    ]
    return filtered_indices_results[parameter_names].to_numpy(dtype=np.float64)
