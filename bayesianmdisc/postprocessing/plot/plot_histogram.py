import math
from typing import Any, Dict, TypeAlias, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from bayesianmdisc.customtypes import NPArray
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.statistics.utility import (
    determine_moments_of_univariate_normal_distribution,
    determine_quantiles_from_samples,
)

TrueParameter: TypeAlias = Union[float, None]
TrueParameters: TypeAlias = tuple[TrueParameter, ...]
FigureLayout = tuple[int, int]
FigureSize = tuple[float, float]


credible_interval = 0.95
cm_to_inch = 1 / 2.54


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"


class HistogramsPlotterConfig:
    def __init__(self) -> None:
        # font sizes
        self.label_size = 7
        # font size in legend
        self.font_size = 7
        self.font: Dict[str, Any] = {"size": self.font_size}
        # figure size
        self.pad_subplots_width = 0.8
        self.pad_subplots_hight = 1.6
        self.additional_height_when_legend_at_bottom = 1.0 * cm_to_inch

        # axis labels
        self.y_axis_label = "probability density [-]"

        # ticks
        self.num_x_ticks = 5
        self.num_y_ticks = 5

        # major ticks
        self.major_tick_label_size = 7
        self.major_ticks_size = 7
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 7
        self.minor_ticks_size = 7
        self.minor_ticks_width = 1

        # histogram
        self.hist_bins = 64  # 128
        self.hist_color = "tab:cyan"
        self.hist_label = "samples"

        # moments
        self.mean_color = "tab:red"
        self.mean_linestyle = "solid"
        self.mean_linewidth = 1.0
        self.mean_label = "mean"

        # truth
        self.truth_color = "tab:orange"
        self.truth_linestyle = "solid"
        self.truth_label = "truth"

        # scientific notation
        self.scientific_notation_size = 6

        # others
        self.max_ratio_of_distance_less_and_greater_mean = 4

        # save options
        self.dpi = 300


def plot_histograms(
    parameter_names: tuple[str, ...],
    true_parameters: TrueParameters,
    samples: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    config = HistogramsPlotterConfig()
    num_parameters = len(parameter_names)
    if num_parameters <= 4:
        num_columns = 2
    else:
        num_columns = 3
    fig_layout, fig_size = _define_figure_layout_and_size(
        num_parameters, num_columns, config
    )

    file_name = f"parameter_histograms.png"
    figure, axes = plt.subplots(fig_layout[0], fig_layout[1], figsize=fig_size)
    axes = axes.flatten()
    figure.tight_layout(
        w_pad=config.pad_subplots_width, h_pad=config.pad_subplots_hight
    )

    def plot_one_histogram(
        parameter_name: str,
        true_parameter: TrueParameter,
        samples: NPArray,
        subplot_index: int,
    ) -> None:
        axis = axes[subplot_index]

        axis.hist(
            samples,
            bins=config.hist_bins,
            density=True,
            color=config.hist_color,
            label=config.hist_label,
        )

        # mean
        moments = determine_moments_of_univariate_normal_distribution(samples)
        mean = moments.mean
        axis.axvline(
            x=mean,
            color=config.mean_color,
            linestyle=config.mean_linestyle,
            linewidth=config.mean_linewidth,
            label=config.mean_label,
        )

        # credible interval
        min_quantile_np, max_quantile_np = determine_quantiles_from_samples(
            samples, credible_interval
        )
        min_quantile = float(min_quantile_np)
        max_quantile = float(max_quantile_np)

        # truth
        if true_parameter is not None:
            axis.axvline(
                x=true_parameter,
                color=config.truth_color,
                linestyle=config.truth_linestyle,
                label=config.truth_label,
            )

        # x axis
        min_sample = float(np.amin(samples))
        max_sample = float(np.amax(samples))
        distance_min_to_mean = mean - min_sample
        distance_mean_to_max = max_sample - mean

        distance_ratio_less_mean_to_greater_mean = (
            distance_mean_to_max / distance_min_to_mean
        )

        if (
            distance_ratio_less_mean_to_greater_mean
            > config.max_ratio_of_distance_less_and_greater_mean
        ):
            max_x = mean + (
                config.max_ratio_of_distance_less_and_greater_mean
                * distance_min_to_mean
            )
            min_x = mean - (1.15 * distance_min_to_mean)
            axis.set_xlim(left=min_x, right=max_x)

        x_ticks = [min_quantile, mean, max_quantile]
        axis.set_xticks(x_ticks)
        axis.xaxis.set_major_formatter(ScalarFormatterForceFormat())

        # y axis
        axis.yaxis.set_major_locator(MaxNLocator(nbins=config.num_y_ticks))
        if subplot_index % num_columns == 0:
            axis.set_ylabel(config.y_axis_label, **config.font)

        # axis
        axis.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axis.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )
        axis.ticklabel_format(
            axis="both",
            style="scientific",
            scilimits=(0, 0),
            useOffset=False,
        )
        axis.yaxis.get_offset_text().set_fontsize(config.scientific_notation_size)
        axis.xaxis.get_offset_text().set_fontsize(config.scientific_notation_size)

        # title
        axis.set_title(parameter_name, **config.font)

    splitted_parameter_samples = _split_samples(samples)
    num_subplots = len(axes)

    for parameter_name, true_parameter, parameter_samples, subplot_index in zip(
        parameter_names,
        true_parameters,
        splitted_parameter_samples,
        range(num_parameters),
    ):
        plot_one_histogram(
            parameter_name, true_parameter, parameter_samples, subplot_index
        )

    for subplot_index in range(num_parameters, num_subplots):
        axes[subplot_index].axis("off")

    # legend
    if _are_all_subfigures_used(num_parameters, num_columns):
        figure.legend(
            *axes[0].get_legend_handles_labels(),
            fontsize=config.font_size,
            loc="outside lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
        )
    else:
        axes[num_parameters - 1].legend(
            fontsize=config.font_size,
            bbox_to_anchor=(1.15, 0.96),
            loc="upper left",
            borderaxespad=0.0,
        )

    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()


def _define_figure_layout_and_size(
    num_parameters: int, num_columns: int, config: HistogramsPlotterConfig
) -> tuple[FigureLayout, FigureSize]:
    width = 16 * cm_to_inch
    height_per_row = 4 * cm_to_inch

    num_rows = int(math.ceil(num_parameters / num_columns))
    figure_layout = (num_rows, num_columns)
    height = num_rows * height_per_row
    if _are_all_subfigures_used(num_parameters, num_columns):
        height += config.additional_height_when_legend_at_bottom
    figure_size = (width, height)
    return figure_layout, figure_size


def _split_samples(samples: NPArray) -> tuple[NPArray, ...]:
    return np.unstack(samples, axis=1)


def _are_all_subfigures_used(num_parameters: int, num_columns: int) -> bool:
    return num_parameters % num_columns == 0
