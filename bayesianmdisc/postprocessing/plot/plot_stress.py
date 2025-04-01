from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from bayesianmdisc.customtypes import Device, NPArray
from bayesianmdisc.data.testcases import test_case_identifier_biaxial_tension
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import LinkaCANN
from bayesianmdisc.statistics.metrics import (
    coefficient_of_determination,
    root_mean_squared_error,
)


class StressPlotterConfigLinkaCANN:
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


def plot_stresses_linka_cann(
    model: LinkaCANN,
    parameter_samples: NPArray,
    inputs: NPArray,
    outputs: NPArray,
    test_cases: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    config = StressPlotterConfigLinkaCANN()
    num_data_sets = 5
    num_points_per_data_set = 11
    min_stretch = 1.0
    max_stretch = 1.1
    data_stretches = np.linspace(min_stretch, max_stretch, num_points_per_data_set)
    stretch_ratios = [(1.0, 1.0), (1.0, 0.75), (0.75, 1.0), (1.0, 0.5), (0.5, 1.0)]
    index_fiber = 0
    index_normal = 1
    num_model_inputs = 512

    def split_inputs_and_outputs(
        inputs: NPArray, outputs: NPArray
    ) -> tuple[list[NPArray], list[NPArray]]:
        input_sets = np.split(inputs, num_data_sets, axis=0)
        output_sets = np.split(outputs, num_data_sets, axis=0)
        return input_sets, output_sets

    def calculate_model_mean_and_stddev(
        model: LinkaCANN,
        parameter_samples: NPArray,
        inputs: NPArray,
        test_cases: NPArray,
    ) -> tuple[NPArray, NPArray]:
        parameter_sample_list = list(
            torch.from_numpy(parameter_samples)
            .type(torch.get_default_dtype())
            .to(device)
        )
        inputs_torch = (
            torch.from_numpy(inputs).type(torch.get_default_dtype()).to(device)
        )
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
        predictions = np.stack(prediction_list, axis=2)
        means = np.mean(predictions, axis=2)
        standard_deviations = np.std(predictions, axis=2)
        return means, standard_deviations

    def calculate_coefficient_of_determinant(
        model: LinkaCANN,
        parameter_samples: NPArray,
        inputs: NPArray,
        test_cases: NPArray,
        outputs: NPArray,
    ) -> float:
        mean_model_outputs, _ = calculate_model_mean_and_stddev(
            model, parameter_samples, inputs, test_cases
        )
        return coefficient_of_determination(mean_model_outputs, outputs)

    def calculate_root_mean_squared_error(
        model: LinkaCANN,
        parameter_samples: NPArray,
        inputs: NPArray,
        test_cases: NPArray,
        outputs: NPArray,
    ) -> float:
        mean_model_outputs, _ = calculate_model_mean_and_stddev(
            model, parameter_samples, inputs, test_cases
        )
        return root_mean_squared_error(mean_model_outputs, outputs)

    def plot_one_input_output_set(
        inputs: NPArray, outputs: NPArray, stretch_ratio: tuple[float, float]
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
            model, parameter_samples, model_inputs, model_test_cases
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
            color=config.model_color_mean_normal_1,
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
            color=config.model_color_mean_normal_2,
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

    input_sets, output_sets = split_inputs_and_outputs(inputs, outputs)

    for input_set, output_set, stretch_ratio in zip(
        input_sets, output_sets, stretch_ratios
    ):
        plot_one_input_output_set(input_set, output_set, stretch_ratio)
