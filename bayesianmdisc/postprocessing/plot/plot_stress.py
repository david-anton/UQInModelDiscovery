from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from bayesianmdisc.customtypes import Device, NPArray
from bayesianmdisc.data.testcases import (
    test_case_identifier_biaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.errors import StressPlotterError
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import IsotropicModelLibrary, ModelProtocol, OrthotropicCANN
from bayesianmdisc.statistics.metrics import (
    coefficient_of_determination,
    root_mean_squared_error,
)


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
        # data
        self.data_label_normal_1 = "data fiber"
        self.data_label_normal_2 = "data normal"
        self.data_marker_normal_1 = "x"
        self.data_marker_normal_2 = "x"
        self.data_color_normal_1 = "tab:blue"
        self.data_color_normal_2 = "tab:purple"
        self.data_marker_size = 5
        # model
        self.model_label_normal_1 = "model fiber"
        self.model_label_normal_2 = "model normal"
        # mean
        self.model_color_mean_normal_1 = "tab:blue"
        self.model_color_mean_normal_2 = "tab:purple"
        # standard deviation
        self.model_stddev_alpha = 0.2
        self.model_color_stddev_normal_1 = "tab:blue"
        self.model_color_stddev_normal_2 = "tab:purple"

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
    num_data_sets = 5
    num_points_per_data_set = 11
    min_stretch = 1.0
    max_stretch = 1.1
    data_stretches = np.linspace(min_stretch, max_stretch, num_points_per_data_set)
    stretch_ratios = [(1.0, 1.0), (1.0, 0.75), (0.75, 1.0), (1.0, 0.5), (0.5, 1.0)]
    output_dim = 2
    index_fiber = 0
    index_normal = 1
    num_model_inputs = 512

    def split_inputs_and_outputs(
        inputs: NPArray, test_cases: NPArray, outputs: NPArray
    ) -> tuple[list[NPArray], list[NPArray], list[NPArray]]:
        input_sets = np.split(inputs, num_data_sets, axis=0)
        output_sets = np.split(outputs, num_data_sets, axis=0)
        test_cases_sets = np.split(test_cases, num_data_sets, axis=0)
        return input_sets, test_cases_sets, output_sets

    def plot_one_input_output_set(
        inputs: NPArray,
        test_cases: NPArray,
        outputs: NPArray,
        stretch_ratio: tuple[float, float],
    ) -> None:
        ratio_fiber = stretch_ratio[index_fiber]
        ratio_normal = stretch_ratio[index_normal]
        file_name = (
            f"stresses_stretch_ratio_fiber_normal_{ratio_fiber}_{ratio_normal}.png"
        )

        figure, axes = plt.subplots()

        # data points
        data_stresses_fiber = outputs[:, index_fiber]
        data_stresses_normal = outputs[:, index_normal]
        axes.plot(
            data_stretches,
            data_stresses_fiber,
            marker=config.data_marker_normal_1,
            color=config.data_color_normal_1,
            markersize=config.data_marker_size,
            linestyle="None",
            label=config.data_label_normal_1,
        )
        axes.plot(
            data_stretches,
            data_stresses_normal,
            marker=config.data_marker_normal_2,
            color=config.data_color_normal_2,
            markersize=config.data_marker_size,
            linestyle="None",
            label=config.data_label_normal_2,
        )

        # model
        min_model_stretches = np.min(inputs, axis=0)
        max_model_stretches = np.max(inputs, axis=0)
        model_inputs = np.linspace(
            min_model_stretches, max_model_stretches, num_model_inputs
        )
        model_test_cases = np.full(
            (num_model_inputs,), test_case_identifier_biaxial_tension
        )
        model_stretches = np.linspace(min_stretch, max_stretch, num_model_inputs)

        means, stddevs = calculate_model_mean_and_stddev(
            model, parameter_samples, model_inputs, model_test_cases, output_dim, device
        )
        means_fiber = means[:, index_fiber]
        means_normal = means[:, index_normal]
        stddevs_fiber = stddevs[:, index_fiber]
        stddevs_normal = stddevs[:, index_normal]

        axes.plot(
            model_stretches,
            means_fiber,
            color=config.model_color_mean_normal_1,
            label=config.model_label_normal_1,
        )
        axes.fill_between(
            model_stretches,
            means_fiber - stddevs_fiber,
            means_fiber + stddevs_fiber,
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
            means_normal - stddevs_normal,
            means_normal + stddevs_normal,
            color=config.model_color_stddev_normal_2,
            alpha=config.model_stddev_alpha,
        )

        # axis ticks
        x_ticks = np.linspace(min_stretch, max_stretch, num=6)
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
        r_squared = calculate_coefficient_of_determinant(
            model, parameter_samples, inputs, test_cases, outputs
        )
        rmse = calculate_root_mean_squared_error(
            model, parameter_samples, inputs, test_cases, outputs
        )
        text = "\n".join(
            (
                r"$R^{2}=%.4f$" % (r_squared,),
                r"$RMSE=%.4f$" % (rmse,),
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

    input_sets, test_cases_sets, output_sets = split_inputs_and_outputs(
        inputs, test_cases, outputs
    )

    for input_set, test_case_set, output_set, stretch_ratio in zip(
        input_sets, output_sets, test_cases_sets, stretch_ratios
    ):
        plot_one_input_output_set(input_set, test_case_set, output_set, stretch_ratio)


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
    output_dim = 1
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
        means, stddevs = calculate_model_mean_and_stddev(
            model,
            parameter_samples,
            model_stretches,
            model_test_cases,
            output_dim,
            device,
        )
        model_stretches_plot = model_stretches.reshape((-1,))
        means_plot = means.reshape((-1,))
        stddevs_plot = stddevs.reshape((-1,))

        axes.plot(
            model_stretches_plot,
            means_plot,
            color=model_color_mean,
            label=model_label,
        )
        axes.fill_between(
            model_stretches_plot,
            means_plot - stddevs_plot,
            means_plot + stddevs_plot,
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
            means_plot - stddevs_plot,
            means_plot + stddevs_plot,
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
        r_squared = calculate_coefficient_of_determinant(
            model,
            parameter_samples,
            inputs,
            metrics_test_cases,
            outputs,
            output_dim,
            device,
        )
        rmse = calculate_root_mean_squared_error(
            model,
            parameter_samples,
            inputs,
            metrics_test_cases,
            outputs,
            output_dim,
            device,
        )
        text = "\n".join(
            (
                r"$R^{2}=%.4f$" % (r_squared,),
                r"$RMSE=%.4f$" % (rmse,),
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
        r_squared = calculate_coefficient_of_determinant(
            model, parameter_samples, inputs, test_cases, outputs, output_dim, device
        )
        rmse = calculate_root_mean_squared_error(
            model, parameter_samples, inputs, test_cases, outputs, output_dim, device
        )
        text = "\n".join(
            (
                r"$R^{2}=%.4f$" % (r_squared,),
                r"$RMSE=%.4f$" % (rmse,),
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
        inputs, outputs, test_cases
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
    output_dim = 2
    num_model_inputs_per_set = 128

    def split_inputs_and_outputs(
        inputs: NPArray, test_cases: NPArray, outputs: NPArray
    ) -> tuple[list[NPArray], list[NPArray], list[NPArray]]:

        def validate_data() -> None:
            num_inputs = len(inputs)
            num_outputs = len(outputs)
            num_test_cases = len(test_cases)
            valid_data = (
                num_inputs == num_data_points
                and num_outputs == num_data_points
                and num_test_cases == num_data_points
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

            data_stretch_1 = data_stretches[0, 0]
            data_stretches_2 = data_stretches[:, 1]
            min_stretch_2 = np.amin(data_stretches_2)
            max_stretch_2 = np.amax(data_stretches_2)

            color = colors[set_index]

            # data points
            axes.plot(
                data_stretches_2,
                data_stresses,
                marker=config.data_marker,
                color=color,
                markersize=config.data_marker_size,
                linestyle="None",
                label=config.data_label + f"{data_stretch_1:.2f}",
            )

            # model
            model_stretches_1 = np.full((num_model_inputs_per_set, 1), data_stretch_1)
            model_stretches_2 = np.linspace(
                min_stretch_2, max_stretch_2, num_model_inputs_per_set
            ).reshape((-1, 1))
            model_stretches = np.hstack((model_stretches_1, model_stretches_2))
            model_test_cases = np.full((num_model_inputs_per_set,), test_case)

            means, stddevs = calculate_model_mean_and_stddev(
                model,
                parameter_samples,
                model_stretches,
                model_test_cases,
                output_dim,
                device,
            )
            model_stretches_plot = model_stretches_2
            means_plot = means[:, stress_dim]
            stddevs_plot = stddevs[:, stress_dim]

            axes.plot(
                model_stretches_plot,
                means_plot,
                color=color,
            )
            axes.fill_between(
                model_stretches_plot,
                means_plot - stddevs_plot,
                means_plot + stddevs_plot,
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
                y_label = r"$P_{11}" + "[kPa]"
            else:
                y_label = r"$P_{22}" + "[kPa]"

            axes.set_xlabel(r"$\lambda_{2}$" + "[-]", **config.font)
            axes.set_ylabel(ylabel=y_label, **config.font)
            axes.tick_params(
                axis="both", which="minor", labelsize=config.minor_tick_label_size
            )
            axes.tick_params(
                axis="both", which="major", labelsize=config.major_tick_label_size
            )

            # legend
            axes.legend(fontsize=config.font_size, loc="outside upper right")

            # text box metrics
            r_squared = calculate_coefficient_of_determinant(
                model,
                parameter_samples,
                input_set,
                test_case_set,
                output_set,
                output_dim,
                device,
            )
            rmse = calculate_root_mean_squared_error(
                model,
                parameter_samples,
                input_set,
                test_case_set,
                output_set,
                output_dim,
                device,
            )
            text = "\n".join(
                (
                    r"$R^{2}=%.4f$" % (r_squared,),
                    r"$RMSE=%.4f$" % (rmse,),
                )
            )
            text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
            axes.text(
                0.03,
                0.97,
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


def calculate_model_mean_and_stddev(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    output_dim: int,
    device: Device,
) -> tuple[NPArray, NPArray]:
    parameter_sample_list = list(
        torch.from_numpy(parameter_samples).type(torch.get_default_dtype()).to(device)
    )
    inputs_torch = torch.from_numpy(inputs).type(torch.get_default_dtype()).to(device)
    test_cases_torch = (
        torch.from_numpy(test_cases).type(torch.get_default_dtype()).to(device)
    )

    prediction_list = []
    for parameter_sample in parameter_sample_list:
        prediction_list += [
            model(inputs_torch, test_cases_torch, parameter_sample)
            .cpu()
            .detach()
            .numpy()
        ]
    if output_dim == 2:
        predictions = np.stack(prediction_list, axis=2)
        means = np.mean(predictions, axis=2)
        standard_deviations = np.std(predictions, axis=2)
    else:
        predictions = np.stack(prediction_list, axis=1)
        means = np.mean(predictions, axis=1)
        standard_deviations = np.std(predictions, axis=1)
    return means, standard_deviations


def calculate_coefficient_of_determinant(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    output_dim: int,
    device: Device,
) -> float:
    mean_model_outputs, _ = calculate_model_mean_and_stddev(
        model, parameter_samples, inputs, test_cases, output_dim, device
    )
    return coefficient_of_determination(mean_model_outputs, outputs)


def calculate_root_mean_squared_error(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    output_dim: int,
    device: Device,
) -> float:
    mean_model_outputs, _ = calculate_model_mean_and_stddev(
        model, parameter_samples, inputs, test_cases, output_dim, device
    )
    return root_mean_squared_error(mean_model_outputs, outputs)
