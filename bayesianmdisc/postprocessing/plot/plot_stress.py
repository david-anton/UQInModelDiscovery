from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from torch import vmap

from bayesianmdisc.customtypes import Device, NPArray
from bayesianmdisc.data.testcases import (
    test_case_identifier_biaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
    test_case_identifier_simple_shear,
)
from bayesianmdisc.errors import StressPlotterError
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import IsotropicModelLibrary, ModelProtocol, OrthotropicCANN
from bayesianmdisc.statistics.metrics import (
    coefficient_of_determination,
    coverage_test,
    root_mean_squared_error,
)
from bayesianmdisc.statistics.utility import determine_quantiles

credible_interval = 0.95
factor_stddevs = 1.96  # corresponds to 95%-credible interval


class StressPlotterConfigTreloar:
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

        ### stresses
        # data
        self.data_label_ut = "data ut"
        self.data_label_ebt = "data ebt"
        self.data_label_ps = "data ps"
        self.data_marker_ut = "x"
        self.data_marker_ebt = "x"
        self.data_marker_ps = "x"
        self.data_color_ut = "tab:blue"
        self.data_color_ebt = "tab:purple"
        self.data_color_ps = "tab:green"
        self.data_marker_size = 5
        # model
        self.model_label_ut = "model ut"
        self.model_label_ebt = "model ebt"
        self.model_label_ps = "model ps"
        # mean
        self.model_color_mean_ut = "tab:blue"
        self.model_color_mean_ebt = "tab:purple"
        self.model_color_mean_ps = "tab:green"
        # standard deviation
        self.model_stddev_alpha = 0.2
        self.model_color_stddev_ut = "tab:blue"
        self.model_color_stddev_ebt = "tab:purple"
        self.model_color_stddev_ps = "tab:green"

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_stresses_treloar(
    model: IsotropicModelLibrary,
    parameter_samples: NPArray,
    inputs: NPArray,
    outputs: NPArray,
    test_cases: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    config = StressPlotterConfigTreloar()
    considered_test_cases = [
        test_case_identifier_uniaxial_tension,
        test_case_identifier_equibiaxial_tension,
        test_case_identifier_pure_shear,
    ]
    num_data_points_ut = 25
    num_data_points_ebt = 14
    num_data_points_ps = 14
    expected_set_sizes = [num_data_points_ut, num_data_points_ebt, num_data_points_ps]
    num_model_inputs = 256

    figure_all, axes_all = plt.subplots()

    def split_inputs_and_outputs(
        inputs: NPArray, test_cases: NPArray, outputs: NPArray
    ) -> tuple[list[NPArray], list[int], list[NPArray]]:

        def split_data() -> tuple[list[NPArray], list[int], list[NPArray]]:
            input_sets = []
            test_case_sets = []
            output_sets = []

            for test_case in considered_test_cases:
                filter = test_cases == test_case
                input_sets += [inputs[filter]]
                test_case_sets += [test_case]
                output_sets += [outputs[filter]]

            return input_sets, test_case_sets, output_sets

        def validate_data_sets(
            input_sets: list[NPArray],
            test_case_sets: list[int],
            output_sets: list[NPArray],
        ) -> None:
            input_set_sizes = [len(set) for set in input_sets]
            output_set_sizes = [len(set) for set in output_sets]

            valid_set_sizes = (
                input_set_sizes == expected_set_sizes
                and output_set_sizes == expected_set_sizes
                and len(test_case_sets) == 3
            )
            valid_test_case_sets = test_case_sets == considered_test_cases

            if not valid_set_sizes and valid_test_case_sets:
                raise StressPlotterError(
                    """The number of data points to be plotted
                                        does not match the size of the Treloar data set."""
                )

        input_sets, test_case_sets, output_sets = split_data()
        validate_data_sets(input_sets, test_case_sets, output_sets)
        return input_sets, test_case_sets, output_sets

    def plot_one_input_output_set(
        inputs: NPArray, test_case: int, outputs: NPArray
    ) -> None:
        if test_case == test_case_identifier_uniaxial_tension:
            test_case_label = "uniaxial_tension"
            data_marker = config.data_marker_ut
            data_color = config.data_color_ut
            data_label = config.data_label_ut
            model_color_mean = config.model_color_mean_ut
            model_color_stddev = config.model_color_stddev_ut
            model_label = config.model_label_ut
        elif test_case == test_case_identifier_equibiaxial_tension:
            test_case_label = "equibiaxial_tension"
            data_marker = config.data_marker_ebt
            data_color = config.data_color_ebt
            data_label = config.data_label_ebt
            model_color_mean = config.model_color_mean_ebt
            model_color_stddev = config.model_color_stddev_ebt
            model_label = config.model_label_ebt
        else:
            test_case_label = "pure_shear"
            data_marker = config.data_marker_ps
            data_color = config.data_color_ps
            data_label = config.data_label_ps
            model_color_mean = config.model_color_mean_ps
            model_color_stddev = config.model_color_stddev_ps
            model_label = config.model_label_ps

        file_name = f"treloar_data_{test_case_label}.png"

        figure, axes = plt.subplots()

        data_stretches = inputs[:, 0]
        data_stresses = outputs
        min_stretch = np.amin(data_stretches)
        max_stretch = np.amax(data_stretches)

        # data points
        axes.plot(
            data_stretches,
            data_stresses,
            marker=data_marker,
            color=data_color,
            markersize=config.data_marker_size,
            linestyle="None",
            label=data_label,
        )
        axes_all.plot(
            data_stretches,
            data_stresses,
            marker=data_marker,
            color=data_color,
            markersize=config.data_marker_size,
            linestyle="None",
            label=data_label,
        )

        # model
        model_stretches = np.linspace(
            min_stretch, max_stretch, num_model_inputs
        ).reshape((-1, 1))
        model_test_cases = np.full((num_model_inputs,), test_case)
        means, _ = calculate_model_mean_and_stddev(
            model,
            parameter_samples,
            model_stretches,
            model_test_cases,
            device,
        )
        min_quantiles, max_quantiles = calculate_model_quantiles(
            model, parameter_samples, model_stretches, model_test_cases, device
        )

        model_stretches_plot = model_stretches.reshape((-1,))
        means_plot = means.reshape((-1,))
        min_quantiles_plot = min_quantiles.reshape((-1,))
        max_quantiles_plot = max_quantiles.reshape((-1,))

        axes.plot(
            model_stretches_plot,
            means_plot,
            color=model_color_mean,
            label=model_label,
        )
        axes.fill_between(
            model_stretches_plot,
            min_quantiles_plot,
            max_quantiles_plot,
            color=model_color_stddev,
            alpha=config.model_stddev_alpha,
        )
        axes_all.plot(
            model_stretches_plot,
            means_plot,
            color=model_color_mean,
            label=model_label,
        )
        axes_all.fill_between(
            model_stretches_plot,
            min_quantiles_plot,
            max_quantiles_plot,
            color=model_color_stddev,
            alpha=config.model_stddev_alpha,
        )

        # axis ticks
        x_ticks = np.linspace(min_stretch, max_stretch, num=6)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_tick_labels)

        # axis labels
        axes.set_xlabel("stretch [-]", **config.font)
        axes.set_ylabel("stress [kPa]", **config.font)
        axes.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axes.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )

        # legend
        axes.legend(fontsize=config.font_size, loc="upper left")

        # text box metrics
        num_data_inputs = len(inputs)
        metrics_test_cases = np.full((num_data_inputs,), test_case)
        coverage = calclulate_coverage(
            model,
            parameter_samples,
            inputs,
            metrics_test_cases,
            outputs,
            device,
        )
        r_squared = calculate_coefficient_of_determinant(
            model,
            parameter_samples,
            inputs,
            metrics_test_cases,
            outputs,
            device,
        )
        rmse = calculate_root_mean_squared_error(
            model,
            parameter_samples,
            inputs,
            metrics_test_cases,
            outputs,
            device,
        )
        text = "\n".join(
            (
                r"$C_{95\%}=$" + r"${0}\%$".format(round(coverage, 2)),
                r"$R^{2}=$" + r"${0}$".format(round(r_squared, 4)),
                r"$RMSE=$" + r"${0}$".format(round(rmse, 4)),
            )
        )
        text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
        axes.text(
            0.03,
            0.8,
            text,
            transform=axes.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

        output_path = project_directory.create_output_file_path(
            file_name=file_name, subdir_name=output_subdirectory
        )

        figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)

    def plot_all_input_and_output_sets() -> None:
        file_name = f"treloar_data.png"

        # data
        data_stretches = inputs[:, 0]
        min_stretch = np.amin(data_stretches)
        max_stretch = np.amax(data_stretches)

        # axis ticks
        x_ticks = np.linspace(min_stretch, max_stretch, num=6)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axes_all.set_xticks(x_ticks)
        axes_all.set_xticklabels(x_tick_labels)

        # axis labels
        axes_all.set_xlabel("stretch [-]", **config.font)
        axes_all.set_ylabel("stress [kPa]", **config.font)
        axes_all.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axes_all.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )

        # legend
        axes_all.legend(fontsize=config.font_size, loc="upper left")

        output_path = project_directory.create_output_file_path(
            file_name=file_name, subdir_name=output_subdirectory
        )
        figure_all.savefig(output_path, bbox_inches="tight", dpi=config.dpi)

        # text box metrics
        coverage = calclulate_coverage(
            model,
            parameter_samples,
            inputs,
            test_cases,
            outputs,
            device,
        )
        r_squared = calculate_coefficient_of_determinant(
            model, parameter_samples, inputs, test_cases, outputs, device
        )
        rmse = calculate_root_mean_squared_error(
            model, parameter_samples, inputs, test_cases, outputs, device
        )
        text = "\n".join(
            (
                r"$C_{95\%}=$" + r"${0}\%$".format(round(coverage, 2)),
                r"$R^{2}=$" + r"${0}$".format(round(r_squared, 4)),
                r"$RMSE=$" + r"${0}$".format(round(rmse, 4)),
            )
        )
        text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
        axes_all.text(
            0.45,
            0.05,
            text,
            transform=axes_all.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

    input_sets, test_case_sets, output_sets = split_inputs_and_outputs(
        inputs, test_cases, outputs
    )

    for input_set, test_case, output_set in zip(
        input_sets, test_case_sets, output_sets
    ):
        plot_one_input_output_set(input_set, test_case, output_set)

    plot_all_input_and_output_sets()
    plt.clf()


class StressPlotterConfigKawabata:
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

        ### stresses
        # color map
        self.color_map = "viridis"
        # data
        self.data_label = r"$\lambda_{1}=$"
        self.data_marker = "o"
        self.data_marker_size = 5
        # model
        self.model_label = "model"
        # standard deviation
        self.model_stddev_alpha = 0.2

        # legend
        self.legend_color = "gray"

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_stresses_kawabata(
    model: IsotropicModelLibrary,
    parameter_samples: NPArray,
    inputs: NPArray,
    outputs: NPArray,
    test_cases: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    config = StressPlotterConfigKawabata()
    num_data_points = 76
    set_sizes = [1, 6, 6, 8, 8, 8, 8, 8, 7, 7, 6, 3]
    num_sets = len(set_sizes)
    num_model_inputs_per_set = 128

    stretches_2 = inputs[:, 1]
    min_stretch_2 = np.amin(stretches_2)
    max_stretch_2 = np.amax(stretches_2)

    def split_inputs_and_outputs(
        inputs: NPArray, test_cases: NPArray, outputs: NPArray
    ) -> tuple[list[NPArray], list[NPArray], list[NPArray]]:

        def validate_data() -> None:
            num_inputs = len(inputs)
            num_test_cases = len(test_cases)
            num_outputs = len(outputs)
            valid_data = (
                num_inputs == num_data_points
                and num_test_cases == num_data_points
                and num_outputs == num_data_points
            )

            if not valid_data:
                raise StressPlotterError(
                    f"""The input and/or output do not comprise the expected number of data points 
                    (input comprises {num_inputs} points, test cases {num_test_cases} points and 
                    output {num_outputs} but {num_data_points} data points are expected)."""
                )

        def determine_split_indices() -> list[int]:
            split_indices = [set_sizes[0]]
            for i in range(1, num_sets):
                split_indices += [split_indices[-1] + set_sizes[i]]
            return split_indices

        def split_data_sets(
            split_indices: list[int],
        ) -> tuple[list[NPArray], list[NPArray], list[NPArray]]:
            input_sets = np.split(inputs, split_indices)
            test_case_sets = np.split(test_cases, split_indices)
            output_sets = np.split(outputs, split_indices)
            return input_sets, test_case_sets, output_sets

        validate_data()
        split_indices = determine_split_indices()
        return split_data_sets(split_indices)

    def plot_one_input_output_set(
        input_sets: list[NPArray],
        output_sets: list[NPArray],
        test_case_sets: list[NPArray],
        stress_dim: int,
    ) -> None:
        file_name = f"kawabata_data_stress_{stress_dim}.png"

        figure, axes = plt.subplots()
        color_map = plt.get_cmap(config.color_map, num_sets)
        colors = color_map(np.linspace(0, 1, num_sets))

        for set_index, input_set, test_case_set, output_set in zip(
            range(num_sets), input_sets, test_case_sets, output_sets
        ):
            data_stretches = input_set
            test_case = test_case_set[0]
            data_stresses = output_set[:, stress_dim]

            data_set_stretch_1 = data_stretches[0, 0]
            data_set_stretches_2 = data_stretches[:, 1]
            min_data_set_stretch_2 = np.amin(data_set_stretches_2)
            max_data_set_stretch_2 = np.amax(data_set_stretches_2)

            color = colors[set_index]

            # data points
            axes.plot(
                data_set_stretches_2,
                data_stresses,
                marker=config.data_marker,
                color=color,
                markersize=config.data_marker_size,
                linestyle="None",
                label=config.data_label + f"{data_set_stretch_1:.2f}",
            )

            # model
            model_stretches_1 = np.full(
                (num_model_inputs_per_set, 1), data_set_stretch_1
            )
            model_stretches_2 = np.linspace(
                min_data_set_stretch_2, max_data_set_stretch_2, num_model_inputs_per_set
            ).reshape((-1, 1))
            model_stretches = np.hstack((model_stretches_1, model_stretches_2))
            model_test_cases = np.full((num_model_inputs_per_set,), test_case)

            means, _ = calculate_model_mean_and_stddev(
                model,
                parameter_samples,
                model_stretches,
                model_test_cases,
                device,
                stress_dim,
            )
            min_quantiles, max_quantiles = calculate_model_quantiles(
                model,
                parameter_samples,
                model_stretches,
                model_test_cases,
                device,
                stress_dim,
            )
            model_stretches_plot = model_stretches_2.reshape((-1,))
            means_plot = means.reshape((-1,))
            min_quantiles_plot = min_quantiles.reshape((-1,))
            max_quantiles_plot = max_quantiles.reshape((-1,))

            axes.plot(
                model_stretches_plot,
                means_plot,
                color=color,
            )
            axes.fill_between(
                model_stretches_plot,
                min_quantiles_plot,
                max_quantiles_plot,
                color=color,
                alpha=config.model_stddev_alpha,
            )

        # axis ticks
        x_ticks = np.linspace(min_stretch_2, max_stretch_2, num=6)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_tick_labels)

        # axis labels
        if stress_dim == 0:
            y_label = r"$P_{11}$" + " [kPa]"
        else:
            y_label = r"$P_{22}$" + " [kPa]"

        axes.set_xlabel(r"$\lambda_{2}$" + " [-]", **config.font)
        axes.set_ylabel(ylabel=y_label, **config.font)
        axes.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axes.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )

        # legend
        data_point = Line2D(
            [],
            [],
            color=config.legend_color,
            marker=config.data_marker,
            markersize=config.data_marker_size,
            linestyle="None",
            label="data",
        )
        model_mean = Line2D([], [], color=config.legend_color, label="mean")
        model_stddevs = Patch(
            facecolor=config.legend_color,
            alpha=config.model_stddev_alpha,
            label="95%-credible interval",
        )
        data_legend_handles, _ = axes.get_legend_handles_labels()
        legend_handles = [data_point, model_mean, model_stddevs] + data_legend_handles
        axes.legend(
            handles=legend_handles,
            fontsize=config.font_size,
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )

        # text box metrics
        coverage = calclulate_coverage(
            model,
            parameter_samples,
            inputs,
            test_cases,
            outputs,
            device,
            output_dim=stress_dim,
        )
        r_squared = calculate_coefficient_of_determinant(
            model,
            parameter_samples,
            inputs,
            test_cases,
            outputs,
            device,
            output_dim=stress_dim,
        )
        rmse = calculate_root_mean_squared_error(
            model,
            parameter_samples,
            inputs,
            test_cases,
            outputs,
            device,
            output_dim=stress_dim,
        )
        text = "\n".join(
            (
                r"$C_{95\%}=$" + r"${0}\%$".format(round(coverage, 2)),
                r"$R^{2}=$" + r"${0}$".format(round(r_squared, 4)),
                r"$RMSE=$" + r"${0}$".format(round(rmse, 4)),
            )
        )
        text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
        axes.text(
            0.70,
            0.2,
            text,
            transform=axes.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

        output_path = project_directory.create_output_file_path(
            file_name=file_name, subdir_name=output_subdirectory
        )
        figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)

    input_sets, output_sets, test_case_sets = split_inputs_and_outputs(
        inputs, outputs, test_cases
    )
    plot_one_input_output_set(input_sets, output_sets, test_case_sets, stress_dim=0)
    plot_one_input_output_set(input_sets, output_sets, test_case_sets, stress_dim=1)
    plt.clf()


class StressPlotterConfigLinka:
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

        ### stresses
        # data normal
        self.data_label_normal_1 = "data fiber"
        self.data_label_normal_2 = "data normal"
        self.data_marker_normal_1 = "x"
        self.data_marker_normal_2 = "x"
        self.data_color_normal_1 = "tab:blue"
        self.data_color_normal_2 = "tab:purple"
        self.data_label_shear = "data"
        self.data_marker_shear = "x"
        self.data_color_shear = "tab:blue"
        self.data_marker_size = 5
        # model
        self.model_label_normal_1 = "model fiber"
        self.model_label_normal_2 = "model normal"
        self.model_label_shear = "model"
        # mean
        self.model_color_mean_normal_1 = "tab:blue"
        self.model_color_mean_normal_2 = "tab:purple"
        self.model_color_mean_shear = "tab:blue"
        # standard deviation
        self.model_stddev_alpha = 0.2
        self.model_color_stddev_normal_1 = "tab:blue"
        self.model_color_stddev_normal_2 = "tab:purple"
        self.model_color_stddev_shear = "tab:blue"

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_stresses_linka(
    model: OrthotropicCANN,
    parameter_samples: NPArray,
    inputs: NPArray,
    outputs: NPArray,
    test_cases: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    config = StressPlotterConfigLinka()
    num_data_sets = 11
    num_data_sets_simple_shear = 6
    num_data_sets_biaxial_tension = 5
    num_points_per_data_set = 11
    num_data_points = num_data_sets * num_points_per_data_set
    min_principal_stretch = 1.0
    max_principal_stretch = 1.1
    min_shear_strain = 0.0
    max_shear_strain = 0.5
    data_principal_stretches = np.linspace(
        min_principal_stretch, max_principal_stretch, num_points_per_data_set
    )
    data_shear_strains = np.linspace(
        min_shear_strain, max_shear_strain, num_points_per_data_set
    )

    stretch_ratios = [(1.0, 1.0), (1.0, 0.75), (0.75, 1.0), (1.0, 0.5), (0.5, 1.0)]
    stretch_ratio_index_fiber = 0
    stretch_ratio_index_normal = 1
    stress_labels = ["ff", "fs", "fn", "sf", "ss", "sn", "nf", "ns", "nn"]
    deformation_labels = [None, "sf", "nf", "fs", None, "ns", "fn", "sn", None]
    index_principal_stress_f = 0
    index_shear_stress_fs = 1
    index_shear_stress_fn = 2
    index_shear_stress_sf = 3
    index_shear_stress_sn = 5
    index_shear_stress_nf = 6
    index_shear_stress_ns = 7
    index_principal_stress_n = 8
    shear_indices = [
        index_shear_stress_fs,
        index_shear_stress_fn,
        index_shear_stress_sf,
        index_shear_stress_sn,
        index_shear_stress_nf,
        index_shear_stress_ns,
    ]
    num_model_inputs = 256

    def split_inputs_and_outputs(
        inputs: NPArray, test_cases: NPArray, outputs: NPArray
    ) -> tuple[
        list[NPArray],
        list[NPArray],
        list[NPArray],
        list[NPArray],
        list[NPArray],
        list[NPArray],
    ]:

        def validate_data() -> None:
            num_inputs = len(inputs)
            num_test_cases = len(test_cases)
            num_outputs = len(outputs)
            valid_data = (
                num_inputs == num_data_points
                and num_test_cases == num_data_points
                and num_outputs == num_data_points
            )

            if not valid_data:
                raise StressPlotterError(
                    f"""The input and/or output do not comprise the expected number of data points 
                    (input comprises {num_inputs} points, test cases {num_test_cases} points and 
                    output {num_outputs} but {num_data_points} data points are expected)."""
                )

        def split_data_sets() -> tuple[
            list[NPArray],
            list[NPArray],
            list[NPArray],
            list[NPArray],
            list[NPArray],
            list[NPArray],
        ]:
            input_sets = np.split(inputs, num_data_sets, axis=0)
            test_case_sets = np.split(test_cases, num_data_sets, axis=0)
            output_sets = np.split(outputs, num_data_sets, axis=0)
            input_sets_ss = input_sets[0:num_data_sets_simple_shear]
            test_case_sets_ss = test_case_sets[0:num_data_sets_simple_shear]
            output_sets_ss = output_sets[0:num_data_sets_simple_shear]
            input_sets_bt = input_sets[num_data_sets_simple_shear:]
            test_case_sets_bt = test_case_sets[num_data_sets_simple_shear:]
            output_sets_bt = output_sets[num_data_sets_simple_shear:]
            return (
                input_sets_ss,
                test_case_sets_ss,
                output_sets_ss,
                input_sets_bt,
                test_case_sets_bt,
                output_sets_bt,
            )

        validate_data()
        return split_data_sets()

    def plot_one_simple_shear_set(
        inputs: NPArray, test_cases: NPArray, outputs: NPArray, index: int
    ) -> None:
        label_shear_stress = stress_labels[index]
        label_deformation = deformation_labels[index]
        file_name = f"shearstress_{label_shear_stress}.png"

        figure, axes = plt.subplots()

        # data points
        data_stresses = outputs[:, index]
        axes.plot(
            data_shear_strains,
            data_stresses,
            marker=config.data_marker_shear,
            color=config.data_color_shear,
            markersize=config.data_marker_size,
            linestyle="None",
            label=config.data_label_shear,
        )

        # model
        min_model_inputs = np.min(inputs, axis=0)
        max_model_inputs = np.max(inputs, axis=0)
        model_inputs = np.linspace(min_model_inputs, max_model_inputs, num_model_inputs)
        model_test_cases = np.full(
            (num_model_inputs,), test_case_identifier_simple_shear
        )
        model_strains = np.linspace(
            min_shear_strain, max_shear_strain, num_model_inputs
        )

        means_, _ = calculate_model_mean_and_stddev(
            model, parameter_samples, model_inputs, model_test_cases, device
        )
        min_quantiles, max_quantiles = calculate_model_quantiles(
            model, parameter_samples, model_inputs, model_test_cases, device
        )
        means = means_[:, index]
        min_quantiles = min_quantiles[:, index]
        max_quantiles = max_quantiles[:, index]

        axes.plot(
            model_strains,
            means,
            color=config.model_color_mean_shear,
            label=config.model_label_shear,
        )
        axes.fill_between(
            model_strains,
            min_quantiles,
            max_quantiles,
            color=config.model_color_stddev_shear,
            alpha=config.model_stddev_alpha,
        )

        # axis ticks
        x_ticks = np.linspace(min_shear_strain, max_shear_strain, num=6)
        x_tick_labels = [str(tick) for tick in x_ticks]
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_tick_labels)

        # axis labels
        axes.set_xlabel(
            r"$\gamma_{0}$".format(label_deformation) + " [-]", **config.font
        )
        axes.set_ylabel(
            r"$\sigma_{0}$".format(label_shear_stress) + " [kPa]", **config.font
        )
        axes.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axes.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )

        # legend
        axes.legend(fontsize=config.font_size, loc="upper left")

        # text box metrics
        coverage = calclulate_coverage(
            model,
            parameter_samples,
            inputs,
            test_cases,
            outputs,
            device,
            output_dim=index,
        )
        r_squared = calculate_coefficient_of_determinant(
            model,
            parameter_samples,
            inputs,
            test_cases,
            outputs,
            device,
            output_dim=index,
        )
        rmse = calculate_root_mean_squared_error(
            model,
            parameter_samples,
            inputs,
            test_cases,
            outputs,
            device,
            output_dim=index,
        )
        text = "\n".join(
            (
                r"$C_{95\%}=$" + r"${0}\%$".format(round(coverage, 2)),
                r"$R^{2}=$" + r"${0}$".format(round(r_squared, 4)),
                r"$RMSE=$" + r"${0}$".format(round(rmse, 4)),
            )
        )
        text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
        axes.text(
            0.05,
            0.65,
            text,
            transform=axes.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

        output_path = project_directory.create_output_file_path(
            file_name=file_name, subdir_name=output_subdirectory
        )
        figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
        plt.clf()

    def plot_one_biaxial_tension_set(
        inputs: NPArray,
        test_cases: NPArray,
        outputs: NPArray,
        stretch_ratio: tuple[float, float],
    ) -> None:
        ratio_fiber = stretch_ratio[stretch_ratio_index_fiber]
        ratio_normal = stretch_ratio[stretch_ratio_index_normal]
        file_name = (
            f"principalstresses_stretchratio_f_n_{ratio_fiber}_{ratio_normal}.png"
        )

        figure, axes = plt.subplots()

        # data points
        data_stresses_fiber = outputs[:, index_principal_stress_f]
        data_stresses_normal = outputs[:, index_principal_stress_n]
        axes.plot(
            data_principal_stretches,
            data_stresses_fiber,
            marker=config.data_marker_normal_1,
            color=config.data_color_normal_1,
            markersize=config.data_marker_size,
            linestyle="None",
            label=config.data_label_normal_1,
        )
        axes.plot(
            data_principal_stretches,
            data_stresses_normal,
            marker=config.data_marker_normal_2,
            color=config.data_color_normal_2,
            markersize=config.data_marker_size,
            linestyle="None",
            label=config.data_label_normal_2,
        )

        # model
        min_model_inputs = np.min(inputs, axis=0)
        max_model_inputs = np.max(inputs, axis=0)
        model_inputs = np.linspace(min_model_inputs, max_model_inputs, num_model_inputs)
        model_test_cases = np.full(
            (num_model_inputs,), test_case_identifier_biaxial_tension
        )
        model_stretches = np.linspace(
            min_principal_stretch, max_principal_stretch, num_model_inputs
        )

        means, _ = calculate_model_mean_and_stddev(
            model, parameter_samples, model_inputs, model_test_cases, device
        )
        min_quantiles, max_quantiles = calculate_model_quantiles(
            model, parameter_samples, model_inputs, model_test_cases, device
        )
        means_fiber = means[:, index_principal_stress_f]
        means_normal = means[:, index_principal_stress_n]
        min_quantiles_fiber = min_quantiles[:, index_principal_stress_f]
        max_quantiles_fiber = max_quantiles[:, index_principal_stress_f]
        min_quantiles_normal = min_quantiles[:, index_principal_stress_n]
        max_quantiles_normal = max_quantiles[:, index_principal_stress_n]

        axes.plot(
            model_stretches,
            means_fiber,
            color=config.model_color_mean_normal_1,
            label=config.model_label_normal_1,
        )
        axes.fill_between(
            model_stretches,
            min_quantiles_fiber,
            max_quantiles_fiber,
            color=config.model_color_stddev_normal_1,
            alpha=config.model_stddev_alpha,
        )
        axes.plot(
            model_stretches,
            means_normal,
            color=config.model_color_mean_normal_2,
            label=config.model_label_normal_2,
        )
        axes.fill_between(
            model_stretches,
            min_quantiles_normal,
            max_quantiles_normal,
            color=config.model_color_stddev_normal_2,
            alpha=config.model_stddev_alpha,
        )

        # axis ticks
        x_ticks = np.linspace(min_principal_stretch, max_principal_stretch, num=6)
        x_tick_labels = [str(tick) for tick in x_ticks]
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_tick_labels)

        # axis labels
        axes.set_xlabel("stretch [-]", **config.font)
        axes.set_ylabel("stress [kPa]", **config.font)
        axes.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axes.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )

        # legend
        axes.legend(fontsize=config.font_size, loc="upper left")

        # text box ratios
        text = "\n".join(
            (
                r"$\lambda_{f}=%.2f$" % (ratio_fiber,),
                r"$\lambda_{n}=%.2f$" % (ratio_normal,),
            )
        )
        text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
        axes.text(
            0.45,
            0.95,
            text,
            transform=axes.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

        # text box metrics
        coverage = calclulate_coverage(
            model,
            parameter_samples,
            inputs,
            test_cases,
            outputs,
            device,
        )
        r_squared = calculate_coefficient_of_determinant(
            model, parameter_samples, inputs, test_cases, outputs, device
        )
        rmse = calculate_root_mean_squared_error(
            model, parameter_samples, inputs, test_cases, outputs, device
        )
        text = "\n".join(
            (
                r"$C_{95\%}=$" + r"${0}\%$".format(round(coverage, 2)),
                r"$R^{2}=$" + r"${0}$".format(round(r_squared, 4)),
                r"$RMSE=$" + r"${0}$".format(round(rmse, 4)),
            )
        )
        text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
        axes.text(
            0.05,
            0.65,
            text,
            transform=axes.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

        output_path = project_directory.create_output_file_path(
            file_name=file_name, subdir_name=output_subdirectory
        )
        figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
        plt.clf()

    (
        input_sets_ss,
        test_case_sets_ss,
        output_sets_ss,
        input_sets_bt,
        test_case_sets_bt,
        output_sets_bt,
    ) = split_inputs_and_outputs(inputs, test_cases, outputs)

    for input_set, test_case_set, output_set, index in zip(
        input_sets_ss, test_case_sets_ss, output_sets_ss, shear_indices
    ):
        plot_one_simple_shear_set(input_set, test_case_set, output_set, index)

    for input_set, test_case_set, output_set, stretch_ratio in zip(
        input_sets_bt, test_case_sets_bt, output_sets_bt, stretch_ratios
    ):
        plot_one_biaxial_tension_set(
            input_set, test_case_set, output_set, stretch_ratio
        )


def calculate_model_predictions(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    device: Device,
) -> NPArray:
    parameters_torch = (
        torch.from_numpy(parameter_samples).type(torch.get_default_dtype()).to(device)
    )
    inputs_torch = torch.from_numpy(inputs).type(torch.get_default_dtype()).to(device)
    test_cases_torch = (
        torch.from_numpy(test_cases).type(torch.get_default_dtype()).to(device)
    )

    vmap_func = lambda parameters: model(inputs_torch, test_cases_torch, parameters)
    predictions = vmap(vmap_func)(parameters_torch)
    return predictions.cpu().detach().numpy()


def calculate_model_mean_and_stddev(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> tuple[NPArray, NPArray]:
    predictions = calculate_model_predictions(
        model=model,
        parameter_samples=parameter_samples,
        inputs=inputs,
        test_cases=test_cases,
        device=device,
    )

    means = np.mean(predictions, axis=0)
    standard_deviations = np.std(predictions, axis=0)

    if output_dim is not None:
        means = means[:, output_dim].reshape((-1, 1))
        standard_deviations = standard_deviations[:, output_dim].reshape((-1, 1))

    return means, standard_deviations


def calculate_model_quantiles(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> tuple[NPArray, NPArray]:
    prediction_samples = calculate_model_predictions(
        model=model,
        parameter_samples=parameter_samples,
        inputs=inputs,
        test_cases=test_cases,
        device=device,
    )
    min_quantiles, max_quantiles = determine_quantiles(
        prediction_samples, credible_interval
    )

    if output_dim is not None:
        min_quantiles = min_quantiles[:, output_dim].reshape((-1, 1))
        max_quantiles = max_quantiles[:, output_dim].reshape((-1, 1))

    return min_quantiles, max_quantiles


def calclulate_coverage(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> float:
    prediction_samples = calculate_model_predictions(
        model=model,
        parameter_samples=parameter_samples,
        inputs=inputs,
        test_cases=test_cases,
        device=device,
    )

    if output_dim is not None:
        prediction_samples = prediction_samples[:, :, output_dim]
        outputs = outputs[:, output_dim]

    return coverage_test(prediction_samples, outputs, credible_interval)


def calculate_coefficient_of_determinant(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> float:
    mean_model_outputs, _ = calculate_model_mean_and_stddev(
        model, parameter_samples, inputs, test_cases, device, output_dim
    )
    if output_dim is not None:
        outputs = outputs[:, output_dim].reshape((-1, 1))

    return coefficient_of_determination(mean_model_outputs, outputs)


def calculate_root_mean_squared_error(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> float:
    mean_model_outputs, _ = calculate_model_mean_and_stddev(
        model, parameter_samples, inputs, test_cases, device, output_dim
    )
    if output_dim is not None:
        outputs = outputs[:, output_dim].reshape((-1, 1))

    return root_mean_squared_error(mean_model_outputs, outputs)
