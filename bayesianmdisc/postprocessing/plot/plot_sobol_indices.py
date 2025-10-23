from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from bayesianmdisc.customtypes import NPArray, PDDataFrame
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.readerswriters import CSVDataReader
from bayesianmdisc.postprocessing.plot.utility import split_treloar_inputs_and_outputs
from bayesianmdisc.testcases import (
    map_test_case_identifiers_to_labels,
    test_case_identifier_biaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_23,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_32,
    test_case_identifier_uniaxial_tension,
)

total_indice_label = "total_sobol_indices"
pd_column_lable_test_cases = "test cases"

cm_to_inch = 1 / 2.54


class IndicesDevelopmentPlotterConfigTreloar:
    def __init__(self) -> None:
        # label size
        self.label_size = 7
        # font size in legend
        self.font_size = 7
        self.font: Dict[str, Any] = {"size": self.font_size}
        # figure size
        self.figure_size = (16 * cm_to_inch, 6 * cm_to_inch)
        self.pad_subplots_width = 0.0

        # ticks
        self.num_x_ticks = 5
        self.num_y_ticks = 6

        # major ticks
        self.major_tick_label_size = 7
        self.major_ticks_size = 7
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 7
        self.minor_ticks_size = 7
        self.minor_ticks_width = 1

        # titles
        self.title_ut = "uniaxial tension (UT)"
        self.title_ebt = "equibiaxial tension (EBT)"
        self.title_ps = "pure shear (PS)"

        # labels
        self.xaxis_label = "stretch " + r"$\lambda$" + " [-]"
        self.yaxis_label_total_indice = "total Sobol' indices [-]"

        # results
        self.color_map = "tab10"
        self.marker_list = [
            "o",
            "v",
            "^",
            "s",
            "<",
            ">",
            "p",
            "P",
            "X",
            "H",
            "D",
            "1",
            "2",
            "3",
            "4",
            "8",
            "*",
        ]
        self.marker_size = 4
        self.linewidth = 1.0
        self.linestyle = "dashed"

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300

        # sensitivities
        self.min_total_sobol_indice = 0.0
        self.max_total_sobol_indice = 1.0


def plot_sobol_indice_paths_treloar(
    relevant_parameter_indices: list[int],
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    output_index = 0
    parameter_names, indice_results = read_indices_results(
        indice_label=total_indice_label,
        output_index=output_index,
        relevant_parameter_indices=relevant_parameter_indices,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
    )
    num_relevant_parameters = len(relevant_parameter_indices)

    input_sets, test_case_identifiers, _ = split_treloar_inputs_and_outputs(
        inputs, test_cases, outputs
    )
    input_sets = remove_first_inputs(input_sets)

    config = IndicesDevelopmentPlotterConfigTreloar()
    figure, axes = plt.subplots(1, 3, figsize=config.figure_size, sharey=True)
    figure.tight_layout(w_pad=config.pad_subplots_width)

    def plot_one_indice_development(inputs: NPArray, test_case_identifier: int) -> None:
        if test_case_identifier == test_case_identifier_uniaxial_tension:
            axis = axes[0]
            title = config.title_ut
        elif test_case_identifier == test_case_identifier_equibiaxial_tension:
            axis = axes[1]
            title = config.title_ebt
        else:
            axis = axes[2]
            title = config.title_ps

        results = filter_indice_results_for_test_case(
            indice_results, test_case_identifier, parameter_names
        )

        color_map = plt.get_cmap(config.color_map, num_relevant_parameters)
        colors = color_map(np.linspace(0, 1.0, num_relevant_parameters))

        data_stretches = inputs[:, 0]
        min_stretch = np.amin(data_stretches)
        max_stretch = np.amax(data_stretches)

        for parameter_index in range(num_relevant_parameters):
            color = colors[parameter_index]
            marker = config.marker_list[parameter_index]
            parameter_name_plot = parameter_names[parameter_index]
            results_plot = results[:, parameter_index]
            axis.plot(
                data_stretches,
                results_plot,
                marker=marker,
                color=color,
                markersize=config.marker_size,
                linewidth=config.linewidth,
                linestyle=config.linestyle,
                label=parameter_name_plot,
            )

        # x axis
        x_ticks = np.linspace(min_stretch, max_stretch, num=config.num_x_ticks)
        axis.set_xticks(x_ticks)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axis.set_xticklabels(x_tick_labels)
        axis.set_xlabel(config.xaxis_label, **config.font)

        # y axis
        y_ticks = np.linspace(
            config.min_total_sobol_indice,
            config.max_total_sobol_indice,
            num=config.num_x_ticks,
        )
        axis.set_yticks(y_ticks)
        if test_case_identifier == test_case_identifier_uniaxial_tension:
            y_label = config.yaxis_label_total_indice
            axis.set_ylabel(y_label, **config.font)

        # axis
        axis.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axis.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )

        # title
        axis.set_title(title, **config.font)

    for input_set, test_case_identifier in zip(input_sets, test_case_identifiers):
        plot_one_indice_development(input_set, test_case_identifier)

    # legend
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        fontsize=config.font_size,
        loc="outside lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=6,
    )

    file_name = f"{total_indice_label}.png"
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)


class IndicesDevelopmentPlotterConfigLinka:
    def __init__(self) -> None:
        # label size
        self.label_size = 7
        # font size in legend
        self.font_size = 7
        self.font: Dict[str, Any] = {"size": self.font_size}
        # figure size
        self.figure_size = (16 * cm_to_inch, 20 * cm_to_inch)
        self.pad_subplots_hight = 2.0
        self.pad_subplots_width = -0.5

        # ticks
        self.num_x_ticks = 6
        self.num_y_ticks = 6

        # major ticks
        self.major_tick_label_size = 7
        self.major_ticks_size = 7
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 7
        self.minor_ticks_size = 7
        self.minor_ticks_width = 1

        # titles
        self.title_ps = [
            r"$\sigma_{fs}$",
            r"$\sigma_{fn}$",
            r"$\sigma_{sf}$",
            r"$\sigma_{sn}$",
            r"$\sigma_{nf}$",
            r"$\sigma_{ns}$",
        ]
        self.title_bt_prefix_sigma_ff = r"$\sigma_{ff}$"
        self.title_bt_prefix_sigma_nn = r"$\sigma_{nn}$"
        # self.title_bt_ratios = [
        #     ", ".join(
        #         (
        #             r"$\lambda_{f}=1+%.2f(\lambda-1)$" % (stretch_ratio_fiber,),
        #             r"$\lambda_{n}=1+%.2f(\lambda-1)$" % (stretch_ratio_normal,),
        #         )
        #     )
        #     for stretch_ratio_fiber, stretch_ratio_normal in [
        #         (1.0, 1.0),
        #         (1.0, 0.75),
        #         (0.75, 1.0),
        #         (1.0, 0.5),
        #         (0.5, 1.0),
        #     ]
        # ]
        self.title_bt_ratios = [
            ", ".join(
                (
                    "(" + r"$\lambda_{f}^{*}=%.2f$" % (stretch_ratio_fiber,),
                    r"$\lambda_{n}^{*}=%.2f$" % (stretch_ratio_normal,) + ")",
                )
            )
            for stretch_ratio_fiber, stretch_ratio_normal in [
                (1.0, 1.0),
                (1.0, 0.75),
                (0.75, 1.0),
                (1.0, 0.5),
                (0.5, 1.0),
            ]
        ]

        # labels
        self.xaxis_label_ps = [
            r"$\gamma_{sf}$" + " [-]",
            r"$\gamma_{nf}$" + " [-]",
            r"$\gamma_{fs}$" + " [-]",
            r"$\gamma_{ns}$" + " [-]",
            r"$\gamma_{fn}$" + " [-]",
            r"$\gamma_{sn}$" + " [-]",
        ]
        self.xaxis_label_bt = r"$\lambda$" + " [-]"
        self.yaxis_label_total_indice = "total Sobol' indices [-]"

        # results
        self.color_map = "tab10"
        self.marker_list = [
            "o",
            "v",
            "^",
            "s",
            "<",
            ">",
            "p",
            "P",
            "X",
            "H",
            "D",
            "1",
            "2",
            "3",
            "4",
            "8",
            "*",
        ]
        self.marker_size = 4
        self.linewidth = 1.0
        self.linestyle = "dashed"

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300

        # subfigures
        self.subfigure_indices = [
            [0, 1],
            [1, 1],
            [2, 1],
            [3, 1],
            [4, 1],
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 0],
            [0, 2],
            [1, 2],
            [2, 2],
            [3, 2],
            [4, 2],
        ]
        self.subfigure_indices_with_xaxis = [4, 5, 6, 7, 8, 9, 10, 15]
        self.subfigure_indices_with_yaxis = [5, 6, 7, 8, 9, 10]

        # test cases
        self.test_case_identifiers_ps = [
            test_case_identifier_simple_shear_21,
            test_case_identifier_simple_shear_31,
            test_case_identifier_simple_shear_12,
            test_case_identifier_simple_shear_32,
            test_case_identifier_simple_shear_13,
            test_case_identifier_simple_shear_23,
        ]

        # mechanics
        self.min_principal_stretch = 1.0
        self.max_principal_stretch = 1.1
        self.min_shear_strain = 0.0
        self.max_shear_strain = 0.5

        # sensitivities
        self.min_total_sobol_indice = 0.0
        self.max_total_sobol_indice = 1.0


def plot_sobol_indice_paths_anisotropic(
    relevant_parameter_indices: list[int],
    num_points_per_testcase: int,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:

    config = IndicesDevelopmentPlotterConfigLinka()
    figure, axes = plt.subplots(6, 3, figsize=config.figure_size)
    figure.tight_layout(
        h_pad=config.pad_subplots_hight, w_pad=config.pad_subplots_width
    )
    axes[5, 1].axis("off")
    axes[5, 2].axis("off")
    subfigure_counter = 0

    num_relevant_parameters = len(relevant_parameter_indices)
    color_map = plt.get_cmap(config.color_map, num_relevant_parameters)
    colors = color_map(np.linspace(0, 1.0, num_relevant_parameters))

    def plot_indice_development_for_ps(
        output_index: int, subfigure_counter: int
    ) -> int:
        pure_shear_index = output_index - 1
        parameter_names, results_ = read_indices_results(
            indice_label=total_indice_label,
            output_index=output_index,
            relevant_parameter_indices=relevant_parameter_indices,
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
        )
        results = filter_indice_results_for_test_case(
            indice_results=results_,
            test_case_identifier=config.test_case_identifiers_ps[pure_shear_index],
            parameter_names=parameter_names,
        )

        # axis
        subfigure_indices = config.subfigure_indices[subfigure_counter]
        axis = axes[subfigure_indices[0], subfigure_indices[1]]

        # inputs
        min_input = config.min_shear_strain
        max_input = config.max_shear_strain
        inputs_axis = np.linspace(min_input, max_input, num_points_per_testcase)[1:]

        # plot
        for parameter_index in range(num_relevant_parameters):
            color = colors[parameter_index]
            marker = config.marker_list[parameter_index]
            parameter_name_plot = parameter_names[parameter_index]
            results_plot = results[:, parameter_index]
            axis.plot(
                inputs_axis,
                results_plot,
                marker=marker,
                color=color,
                markersize=config.marker_size,
                linewidth=config.linewidth,
                linestyle=config.linestyle,
                label=parameter_name_plot,
            )

        # x axis
        x_ticks = np.linspace(min_input, max_input, num=config.num_x_ticks)
        axis.set_xticks(x_ticks)
        if subfigure_counter in config.subfigure_indices_with_xaxis:
            x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
            axis.set_xticklabels(x_tick_labels)
            axis.set_xlabel(config.xaxis_label_ps[pure_shear_index], **config.font)
        else:
            axis.set_xticklabels([])

        # y axis
        y_ticks = np.linspace(
            config.min_total_sobol_indice,
            config.max_total_sobol_indice,
            num=config.num_x_ticks,
        )
        axis.set_yticks(y_ticks)
        if subfigure_counter in config.subfigure_indices_with_yaxis:
            y_label = config.yaxis_label_total_indice
            axis.set_ylabel(y_label, **config.font)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=config.num_y_ticks))
        else:
            axis.set_yticklabels([])

        # axis
        axis.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axis.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )

        # title
        axis.set_title(config.title_ps[pure_shear_index], **config.font)

        subfigure_counter += 1
        return subfigure_counter

    def plot_indice_developments_for_bt(
        output_index: int, subfigure_counter: int
    ) -> int:

        def split_indices_results(results: NPArray) -> list[NPArray]:
            return np.split(results, 5, axis=0)

        parameter_names, results_ = read_indices_results(
            indice_label=total_indice_label,
            output_index=output_index,
            relevant_parameter_indices=relevant_parameter_indices,
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
        )
        results = filter_indice_results_for_test_case(
            indice_results=results_,
            test_case_identifier=test_case_identifier_biaxial_tension,
            parameter_names=parameter_names,
        )
        results_sets = split_indices_results(results)

        def plot_one_indice_development_for_bt(
            results: NPArray, ratio_index: int, subfigure_counter: int
        ) -> int:

            # axis
            subfigure_indices = config.subfigure_indices[subfigure_counter]
            axis = axes[subfigure_indices[0], subfigure_indices[1]]

            # inputs
            min_input = config.min_principal_stretch
            max_input = config.max_principal_stretch
            inputs_axis = np.linspace(min_input, max_input, num_points_per_testcase)[1:]

            # plot
            for parameter_index in range(num_relevant_parameters):
                color = colors[parameter_index]
                marker = config.marker_list[parameter_index]
                parameter_name_plot = parameter_names[parameter_index]
                results_plot = results[:, parameter_index]
                axis.plot(
                    inputs_axis,
                    results_plot,
                    marker=marker,
                    color=color,
                    markersize=config.marker_size,
                    linewidth=config.linewidth,
                    linestyle=config.linestyle,
                    label=parameter_name_plot,
                )

            # x axis
            x_ticks = np.linspace(min_input, max_input, num=config.num_x_ticks)
            axis.set_xticks(x_ticks)
            if subfigure_counter in config.subfigure_indices_with_xaxis:
                x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
                axis.set_xticklabels(x_tick_labels)
                axis.set_xlabel(config.xaxis_label_bt, **config.font)
            else:
                axis.set_xticklabels([])

            # y axis
            y_ticks = np.linspace(
                config.min_total_sobol_indice,
                config.max_total_sobol_indice,
                num=config.num_x_ticks,
            )
            axis.set_yticks(y_ticks)
            if subfigure_counter in config.subfigure_indices_with_yaxis:
                y_label = config.yaxis_label_total_indice
                axis.set_ylabel(y_label, **config.font)
                axis.yaxis.set_major_locator(MaxNLocator(nbins=config.num_y_ticks))
            else:
                axis.set_yticklabels([])

            # axis
            axis.tick_params(
                axis="both", which="minor", labelsize=config.minor_tick_label_size
            )
            axis.tick_params(
                axis="both", which="major", labelsize=config.major_tick_label_size
            )

            # title
            if output_index == 0:
                title_stress = config.title_bt_prefix_sigma_ff
            else:
                title_stress = config.title_bt_prefix_sigma_nn

            title_ratio = config.title_bt_ratios[ratio_index]
            title = title_stress + "\n" + title_ratio
            axis.set_title(title, **config.font)

            subfigure_counter += 1
            return subfigure_counter

        for ratio_index, indice_results_set in enumerate(results_sets):
            subfigure_counter = plot_one_indice_development_for_bt(
                indice_results_set, ratio_index, subfigure_counter
            )

        return subfigure_counter

    subfigure_counter = plot_indice_developments_for_bt(
        output_index=0, subfigure_counter=subfigure_counter
    )

    for output_index in range(1, 7):
        subfigure_counter = plot_indice_development_for_ps(
            output_index, subfigure_counter
        )

    subfigure_counter = plot_indice_developments_for_bt(
        output_index=7, subfigure_counter=subfigure_counter
    )

    # legend
    axes[5, 0].legend(
        fontsize=config.font_size,
        bbox_to_anchor=(1.15, 0.96),
        loc="upper left",
        borderaxespad=0.0,
        ncol=2,
    )

    file_name = f"{total_indice_label}.png"
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)


def read_indices_results(
    indice_label: str,
    output_index: int,
    relevant_parameter_indices: list[int],
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> tuple[list[str], PDDataFrame]:
    input_file_name = join_indice_results_file_name(indice_label, output_index)

    return _read_indices_results(
        relevant_parameter_indices=relevant_parameter_indices,
        input_file_name=input_file_name,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
    )


def join_indice_results_file_name(indice_label: str, output_index: int) -> str:
    return f"{indice_label}_output_{output_index}"


def _read_indices_results(
    relevant_parameter_indices: list[int],
    input_file_name: str,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> tuple[list[str], PDDataFrame]:
    data_reader = CSVDataReader(project_directory)

    def filter_relevant_parameter_names(data_frame: PDDataFrame) -> list[str]:
        all_parameter_names = data_frame.columns[1:].tolist()
        return [all_parameter_names[index] for index in relevant_parameter_indices]

    def filter_relevant_results(
        data_frame: PDDataFrame, parameter_names: list[str]
    ) -> PDDataFrame:
        column_labels = [pd_column_lable_test_cases] + parameter_names
        return data_frame[column_labels]

    data_frame = data_reader.read_as_pandas_data_frame(
        input_file_name, output_subdirectory, read_from_output_dir=True
    )
    parameter_names = filter_relevant_parameter_names(data_frame)
    results = filter_relevant_results(data_frame, parameter_names)
    return parameter_names, results


def remove_first_inputs(input_sets: list[NPArray]) -> list[NPArray]:
    return [input_set[1:, :] for input_set in input_sets]


def filter_indice_results_for_test_case(
    indice_results: PDDataFrame, test_case_identifier: int, parameter_names: list[str]
) -> NPArray:
    test_case_label = map_test_case_identifiers_to_labels(
        torch.tensor([test_case_identifier])
    )[0]
    filtered_indice_results = indice_results[
        indice_results[pd_column_lable_test_cases] == test_case_label
    ]
    return filtered_indice_results[parameter_names].to_numpy(dtype=np.float64)


def filter_all_indices_results(
    indices_results: PDDataFrame, parameter_names: list[str]
) -> NPArray:
    return indices_results[parameter_names].to_numpy(dtype=np.float64)
