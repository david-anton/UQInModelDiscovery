from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from bayesianmdisc.customtypes import NPArray, PDDataFrame
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.io.readerswriters import CSVDataReader
from bayesianmdisc.postprocessing.plot.utility import split_treloar_inputs_and_outputs
from bayesianmdisc.testcases import (
    map_test_case_identifiers_to_labels,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)

# first_indices_label = "first_sobol_indices"
total_indice_label = "total_sobol_indices"
# indice_labels = [first_indices_label, total_indices_label]
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
        self.pad_subplots = 0.0

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
        self.title_ut = "uniaxial tension"
        self.title_ebt = "equibiaxial tension"
        self.title_ps = "pure shear"

        # labels
        self.xaxis_label = "stretch [-]"
        self.yaxis_label_total_indice = "total Sobol indice [-]"

        # results
        self.color_map = "tab20b"
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


def plot_sobol_indice_paths_treloar(
    relevant_parameter_indices: list[int],
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    output_index = 0
    parameter_names, indices_results = read_indices_results(
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
    file_name = f"model.png"
    figure, axes = plt.subplots(1, 3, figsize=config.figure_size, sharey=True)
    figure.tight_layout(pad=config.pad_subplots)

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

        results = filter_indices_results_for_test_case(
            indices_results, test_case_identifier, parameter_names
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

        # axis ticks
        x_ticks = np.linspace(min_stretch, max_stretch, num=config.num_x_ticks)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(x_tick_labels)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=config.num_y_ticks))

        # axis labels
        axis.set_xlabel(config.xaxis_label, **config.font)
        if test_case_identifier == test_case_identifier_uniaxial_tension:
            y_label = config.yaxis_label_total_indice
            axis.set_ylabel(y_label, **config.font)
        else:
            axis.yaxis.set_visible(False)
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


# class IndicesDevelopmentPlotterConfigLinka:
#     def __init__(self) -> None:
#         # label size
#         self.label_size = 7
#         # font size in legend
#         self.font_size = 7
#         self.font: Dict[str, Any] = {"size": self.font_size}
#         # figure size
#         self.figure_size = (16 * cm_to_inch, 24 * cm_to_inch)
#         self.pad_subplots = 0.8

#         # ticks
#         self.num_x_ticks = 5
#         self.num_y_ticks = 6

#         # major ticks
#         self.major_tick_label_size = 7
#         self.major_ticks_size = 7
#         self.major_ticks_width = 2

#         # minor ticks
#         self.minor_tick_label_size = 7
#         self.minor_ticks_size = 7
#         self.minor_ticks_width = 1

#         # titles
#         self.title_ps = [
#             r"$\sigma_{fs}$",
#             r"$\sigma_{fn}$",
#             r"$\sigma_{sf}$",
#             r"$\sigma_{sn}$",
#             r"$\sigma_{nf}$",
#             r"$\sigma_{ns}$",
#         ]
#         self.title_bt_prefix_sigma_ff = r"$\sigma_{ff}$"
#         self.title_bt_prefix_sigma_nn = r"$\sigma_{nn}$"
#         self.title_bt_ratios = [
#             "\n".join(
#                 (
#                     r"$\lambda_{f}=1+%.2f(\lambda-1)$" % (stretch_ratio_fiber,),
#                     r"$\lambda_{n}=1+%.2f(\lambda-1)$" % (stretch_ratio_normal,),
#                 )
#             )
#             for stretch_ratio_fiber, stretch_ratio_normal in [
#                 (1.0, 1.0),
#                 (1.0, 0.75),
#                 (0.75, 1.0),
#                 (1.0, 0.5),
#                 (0.5, 1.0),
#             ]
#         ]

#         # labels
#         self.xaxis_label_ps = [
#             r"$\gamma_{sf}$" + " [-]",
#             r"$\gamma_{nf}$" + " [-]",
#             r"$\gamma_{fs}$" + " [-]",
#             r"$\gamma_{ns}$" + " [-]",
#             r"$\gamma_{fn}$" + " [-]",
#             r"$\gamma_{sn}$" + " [-]",
#         ]
#         self.xaxis_label_bt = r"$\lambda$" + " [-]"
#         self.yaxis_label_total_indice = "total Sobol indice [-]"

#         # results
#         self.color_map = "tab20b"
#         self.marker_list = [
#             "o",
#             "v",
#             "^",
#             "s",
#             "<",
#             ">",
#             "p",
#             "P",
#             "X",
#             "H",
#             "D",
#             "1",
#             "2",
#             "3",
#             "4",
#             "8",
#             "*",
#         ]
#         self.marker_size = 4
#         self.linewidth = 1.0
#         self.linestyle = "dashed"

#         # scientific notation
#         self.scientific_notation_size = self.font_size

#         # save options
#         self.dpi = 300


# def plot_sobol_indice_paths_linka(
#     relevant_parameter_indices: list[int],
#     inputs: NPArray,
#     test_cases: NPArray,
#     outputs: NPArray,
#     output_subdirectory: str,
#     project_directory: ProjectDirectory,
# ) -> None:
#     output_index = 0
#     parameter_names, indices_results = read_indices_results(
#         indice_label=total_indice_label,
#         output_index=output_index,
#         relevant_parameter_indices=relevant_parameter_indices,
#         output_subdirectory=output_subdirectory,
#         project_directory=project_directory,
#     )
#     num_relevant_parameters = len(relevant_parameter_indices)

#     input_sets, test_case_identifiers, _ = split_treloar_inputs_and_outputs(
#         inputs, test_cases, outputs
#     )
#     input_sets = remove_first_inputs(input_sets)

#     config = IndicesDevelopmentPlotterConfigTreloar()
#     file_name = f"model.png"
#     figure, axes = plt.subplots(1, 3, figsize=config.figure_size, sharey=True)
#     figure.tight_layout(pad=config.pad_subplots)

#     def plot_one_indice_development(inputs: NPArray, test_case_identifier: int) -> None:
#         if test_case_identifier == test_case_identifier_uniaxial_tension:
#             axis = axes[0]
#             title = config.title_ut
#         elif test_case_identifier == test_case_identifier_equibiaxial_tension:
#             axis = axes[1]
#             title = config.title_ebt
#         else:
#             axis = axes[2]
#             title = config.title_ps

#         results = filter_indices_results_for_test_case(
#             indices_results, test_case_identifier, parameter_names
#         )

#         color_map = plt.get_cmap(config.color_map, num_relevant_parameters)
#         colors = color_map(np.linspace(0, 1.0, num_relevant_parameters))

#         data_stretches = inputs[:, 0]
#         min_stretch = np.amin(data_stretches)
#         max_stretch = np.amax(data_stretches)

#         for parameter_index in range(num_relevant_parameters):
#             color = colors[parameter_index]
#             marker = config.marker_list[parameter_index]
#             parameter_name_plot = parameter_names[parameter_index]
#             results_plot = results[:, parameter_index]
#             axis.plot(
#                 data_stretches,
#                 results_plot,
#                 marker=marker,
#                 color=color,
#                 markersize=config.marker_size,
#                 linewidth=config.linewidth,
#                 linestyle=config.linestyle,
#                 label=parameter_name_plot,
#             )

#         # axis ticks
#         x_ticks = np.linspace(min_stretch, max_stretch, num=config.num_x_ticks)
#         x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
#         axis.set_xticks(x_ticks)
#         axis.set_xticklabels(x_tick_labels)
#         axis.yaxis.set_major_locator(MaxNLocator(nbins=config.num_y_ticks))

#         # axis labels
#         axis.set_xlabel(config.xaxis_label, **config.font)
#         if test_case_identifier == test_case_identifier_uniaxial_tension:
#             y_label = config.yaxis_label_total_indice
#             axis.set_ylabel(y_label, **config.font)
#         else:
#             axis.yaxis.set_visible(False)
#         axis.tick_params(
#             axis="both", which="minor", labelsize=config.minor_tick_label_size
#         )
#         axis.tick_params(
#             axis="both", which="major", labelsize=config.major_tick_label_size
#         )

#         # title
#         axis.set_title(title, **config.font)

#     for input_set, test_case_identifier in zip(input_sets, test_case_identifiers):
#         plot_one_indice_development(input_set, test_case_identifier)

#     # legend
#     figure.legend(
#         *axes[0].get_legend_handles_labels(),
#         fontsize=config.font_size,
#         loc="outside lower center",
#         bbox_to_anchor=(0.5, -0.15),
#         ncol=6,
#     )

#     file_name = f"{total_indice_label}.png"
#     output_path = project_directory.create_output_file_path(
#         file_name=file_name, subdir_name=output_subdirectory
#     )
#     figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)


# class IndicesStatisticsPlotterConfigTreloar:
#     def __init__(self) -> None:
#         # label size
#         self.label_size = 7
#         # font size in legend
#         self.font_size = 7
#         self.font: Dict[str, Any] = {"size": self.font_size}

#         # major ticks
#         self.major_tick_label_size = 12
#         self.major_ticks_size = self.font_size
#         self.major_ticks_width = 2

#         # minor ticks
#         self.minor_tick_label_size = 12
#         self.minor_ticks_size = 12
#         self.minor_ticks_width = 1

#         # labels
#         self.xaxis_label = "Sobol indice [-]"

#         # bars
#         self.bar_width = 0.3
#         self.bar_colors_first_indices = "tab:blue"
#         self.bar_colors_total_indices = "tab:cyan"

#         # errors
#         self.error_bar_color = "black"
#         self.error_bar_cap_size = 4

#         # scientific notation
#         self.scientific_notation_size = self.font_size

#         # save options
#         self.dpi = 300


# def plot_sobol_indice_statistics(
#     relevant_parameter_indices: list[int],
#     num_outputs: int,
#     output_subdirectory: str,
#     project_directory: ProjectDirectory,
# ) -> None:

#     def plot_statistics_for_one_output(output_index: int) -> None:

#         config = IndicesStatisticsPlotterConfigTreloar()

#         parameter_names, first_indices_results_df = read_indices_results(
#             indice_label=first_indices_label,
#             output_index=output_index,
#             relevant_parameter_indices=relevant_parameter_indices,
#             output_subdirectory=output_subdirectory,
#             project_directory=project_directory,
#         )
#         first_indices_results = filter_all_indices_results(
#             first_indices_results_df, parameter_names
#         )
#         parameter_names, total_indices_results_df = read_indices_results(
#             indice_label=total_indices_label,
#             output_index=output_index,
#             relevant_parameter_indices=relevant_parameter_indices,
#             output_subdirectory=output_subdirectory,
#             project_directory=project_directory,
#         )
#         total_indices_results = filter_all_indices_results(
#             total_indices_results_df, parameter_names
#         )

#         # statistics
#         def calculate_statistics(indices_results: NPArray) -> tuple[NPArray, NPArray]:
#             num_results = len(indices_results)
#             means = np.mean(indices_results, axis=0)
#             stder = np.std(indices_results, axis=0) / np.sqrt(num_results)
#             return means, stder

#         first_indices_mean, first_indices_stder = calculate_statistics(
#             first_indices_results
#         )
#         total_indices_mean, total_indices_stder = calculate_statistics(
#             total_indices_results
#         )

#         indices_labels = ["first Sobol indice (S1)", "total Sobol indice (ST)"]
#         colors = [config.bar_colors_first_indices, config.bar_colors_total_indices]
#         indices_means = [first_indices_mean, total_indices_mean]
#         indices_stder = [first_indices_stder, total_indices_stder]

#         figure, axes = plt.subplots()

#         # bars and error bars
#         y = np.arange(len(parameter_names))
#         bar_width = config.bar_width
#         multiplier = 0

#         for index, label in enumerate(indices_labels):
#             means = indices_means[index]
#             stder = indices_stder[index]

#             offset = bar_width * multiplier
#             y_with_offset = y + offset
#             color = colors[index]

#             axes.barh(
#                 y_with_offset,
#                 means,
#                 color=color,
#                 height=bar_width,
#                 align="center",
#                 label=label,
#             )
#             if index == 1:
#                 axes.errorbar(
#                     means,
#                     y_with_offset,
#                     xerr=stder,
#                     ecolor=config.error_bar_color,
#                     capsize=config.error_bar_cap_size,
#                     linestyle="None",
#                     label="standard error",
#                 )
#             else:
#                 axes.errorbar(
#                     means,
#                     y_with_offset,
#                     xerr=stder,
#                     ecolor=config.error_bar_color,
#                     capsize=config.error_bar_cap_size,
#                     linestyle="None",
#                 )

#             multiplier += 1

#         # legend
#         axes.legend(
#             fontsize=config.font_size,
#             loc="center right",
#         )

#         # axis labels
#         axes.set_xlabel(config.xaxis_label, **config.font)
#         yticks = y + offset / 2
#         axes.set_yticks(yticks, parameter_names)

#         # saving
#         file_name = f"sobol_indices_statistics_output_{output_index}.png"
#         output_path = project_directory.create_output_file_path(
#             file_name=file_name, subdir_name=output_subdirectory
#         )
#         figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)

#     for output_index in range(num_outputs):
#         plot_statistics_for_one_output(output_index)


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
        relevant_parameter_names = [
            all_parameter_names[index] for index in relevant_parameter_indices
        ]
        return relevant_parameter_names

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


def filter_all_indices_results(
    indices_results: PDDataFrame, parameter_names: list[str]
) -> NPArray:
    return indices_results[parameter_names].to_numpy(dtype=np.float64)
