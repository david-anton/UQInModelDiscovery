from typing import Any, Dict, TypeAlias, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scipy.stats

from bayesianmdisc.customtypes import NPArray
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.statistics.utility import (
    MomentsMultivariateNormal,
    MomentsUnivariateNormal,
    determine_quantiles,
)

TrueParameter: TypeAlias = Union[float, None]
TrueParametersTuple: TypeAlias = tuple[TrueParameter, ...]


credible_interval = 0.95

cm_in_inches = 1 / 2.54  # centimeters in inches


class UnivariateNormalPlotterConfig:
    def __init__(self) -> None:
        # font sizes
        self.label_size = 14
        # font size in legend
        self.font_size = 14
        self.font: Dict[str, Any] = {"size": self.font_size}

        # title pad
        self.title_pad = 10

        # truth
        self.truth_color = "tab:orange"
        self.truth_linestyle = "solid"

        # histogram
        self.hist_bins = 128
        self.hist_color = "tab:cyan"

        # moments
        self.mean_color = "tab:red"
        self.mean_linestyle = "solid"

        # kde
        self.kde_color = "tab:blue"
        self.kde_linestyle = "solid"

        # major ticks
        self.major_tick_label_size = 12
        self.major_ticks_size = self.font_size
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 12
        self.minor_ticks_size = 12
        self.minor_ticks_width = 1

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300
        self.figure_size = (16 * cm_in_inches, 12 * cm_in_inches)
        self.file_format = "pdf"


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"


def plot_histograms(
    parameter_names: tuple[str, ...],
    true_parameters: TrueParametersTuple,
    moments: MomentsMultivariateNormal,
    samples: NPArray,
    algorithm_name: str,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    num_parameters = len(parameter_names)
    if num_parameters == 1:
        parameter_name = parameter_names[0]
        true_parameter = true_parameters[0]
        means = moments.mean
        if moments.covariance.ndim == 2:
            covariance = moments.covariance[0]
        else:
            covariance = moments.covariance
        mean_univariate = means[0]
        std_univariate = np.sqrt(covariance[0])
        moments_univariate = MomentsUnivariateNormal(
            mean=mean_univariate, standard_deviation=std_univariate
        )
        config = UnivariateNormalPlotterConfig()
        plot_univariate_distribution(
            parameter_name,
            true_parameter,
            moments_univariate,
            samples,
            algorithm_name,
            output_subdirectory,
            project_directory,
            config,
        )
    else:
        plot_multivariate_distribution(
            parameter_names,
            true_parameters,
            moments,
            samples,
            algorithm_name,
            output_subdirectory,
            project_directory,
        )


def plot_multivariate_distribution(
    parameter_names: tuple[str, ...],
    true_parameters: TrueParametersTuple,
    moments: MomentsMultivariateNormal,
    samples: NPArray,
    algorithm_name: str,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    num_parameters = len(parameter_names)
    means = moments.mean
    covariance = moments.covariance

    for parameter_idx in range(num_parameters):
        parameter_name = parameter_names[parameter_idx]
        true_parameter = true_parameters[parameter_idx]
        mean_univariate = means[parameter_idx]
        std_univariate = np.sqrt(covariance[parameter_idx, parameter_idx])
        moments_univariate = MomentsUnivariateNormal(
            mean=mean_univariate, standard_deviation=std_univariate
        )
        samples_univariate = samples[:, parameter_idx]
        config = UnivariateNormalPlotterConfig()
        plot_univariate_distribution(
            parameter_name,
            true_parameter,
            moments_univariate,
            samples_univariate,
            algorithm_name,
            output_subdirectory,
            project_directory,
            config,
        )


def plot_univariate_distribution(
    parameter_name: str,
    true_parameter: TrueParameter,
    moments: MomentsUnivariateNormal,
    samples: NPArray,
    algorithm_name: str,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    config: UnivariateNormalPlotterConfig,
) -> None:
    _plot_univariate_distribution_histogram(
        parameter_name=parameter_name,
        true_parameter=true_parameter,
        moments=moments,
        samples=samples,
        algorithm_name=algorithm_name,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
        config=config,
    )


def _plot_univariate_distribution_histogram(
    parameter_name: str,
    true_parameter: TrueParameter,
    moments: MomentsUnivariateNormal,
    samples: NPArray,
    algorithm_name: str,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    config: UnivariateNormalPlotterConfig,
) -> None:
    title = "Posterior probability density"
    mean = moments.mean
    min_quantile_np, max_quantile_np = determine_quantiles(samples, credible_interval)
    min_quantile = float(min_quantile_np)
    max_quantile = float(max_quantile_np)

    figure, axes = plt.subplots(figsize=config.figure_size)
    # Truth
    if true_parameter is not None:
        axes.axvline(
            x=true_parameter,
            color=config.truth_color,
            linestyle=config.truth_linestyle,
            label="truth",
        )

    # Histogram
    _, bin_edges, _ = axes.hist(
        samples,
        bins=config.hist_bins,
        density=True,
        color=config.hist_color,
        label="samples",
    )

    # KDE
    min_bin_edge = np.amin(bin_edges)
    max_bin_edge = np.amax(bin_edges)
    x = np.linspace(start=min_bin_edge, stop=max_bin_edge, num=1024, endpoint=True)
    kde = scipy.stats.gaussian_kde(dataset=samples, bw_method="scott")
    kde_bandwith = kde.factor
    y = kde.pdf(x)
    axes.plot(
        x,
        y,
        color=config.kde_color,
        linestyle=config.kde_linestyle,
        label=f"KDE (bw={round(kde_bandwith,2)})",
    )

    # Ticks
    x_ticks = [min_quantile, mean, max_quantile]
    axes.axvline(
        x=mean,
        color=config.mean_color,
        linestyle=config.mean_linestyle,
        label="mean",
    )

    axes.set_xticks(x_ticks)
    axes.set_title(title, pad=config.title_pad, **config.font)
    axes.legend(fontsize=config.font_size, loc="best")
    axes.set_xlabel(parameter_name, **config.font)
    axes.set_ylabel("probability density", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    axes.xaxis.set_major_formatter(ScalarFormatterForceFormat())
    axes.ticklabel_format(
        axis="both",
        style="scientific",
        scilimits=(0, 0),
        useOffset=False,
    )
    file_name = f"histogram_{parameter_name}_{algorithm_name}.{config.file_format}"
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdirectory
    )
    figure.savefig(
        output_path, format=config.file_format, dpi=config.dpi
    )  # bbox_inches="tight"
    plt.close()
