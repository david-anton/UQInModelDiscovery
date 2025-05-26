from typing import Any, Dict, Optional, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from torch import vmap

from bayesianmdisc.customtypes import Device, NPArray, Tensor
from bayesianmdisc.data.testcases import (
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.errors import PlotterError
from bayesianmdisc.gps.base import GPMultivariateNormal
from bayesianmdisc.gps.gp import GP
from bayesianmdisc.gps.multioutputgp import IndependentMultiOutputGP
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import IsotropicModelLibrary, ModelProtocol, OrthotropicCANN
from bayesianmdisc.models.base import assemble_stretches_from_factors
from bayesianmdisc.statistics.metrics import (
    coefficient_of_determination,
    coverage_test,
    root_mean_squared_error,
)
from bayesianmdisc.postprocessing.plot.utility import (
    split_treloar_inputs_and_outputs,
    split_linka_inputs_and_outputs,
    split_kawabata_inputs_and_outputs,
)
from bayesianmdisc.statistics.utility import determine_quantiles

GaussianProcess: TypeAlias = GP | IndependentMultiOutputGP

credible_interval = 0.95
factor_stddev_credible_interval = 1.96


################################################################################
# Model
################################################################################


class ModelStressPlotterConfigTreloar:
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
        self.model_color_ut = "tab:blue"
        self.model_color_ebt = "tab:purple"
        self.model_color_ps = "tab:green"
        # mean
        self.model_label_mean_ut = "mean ut"
        self.model_label_mean_ebt = "mean ebt"
        self.model_label_mean_ps = "mean ps"
        self.model_mean_linewidth = 1.0
        # credible interval
        self.model_credible_interval_alpha = 0.4
        # samples
        self.model_label_samples_ut = "samples ut"
        self.model_label_samples_ebt = "samples ebt"
        self.model_label_samples_ps = "samples ps"
        self.model_samples_color = "tab:gray"
        self.model_samples_linewidth = 1.0
        self.model_samples_alpha = 0.2

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_model_stresses_treloar(
    model: IsotropicModelLibrary,
    parameter_samples: NPArray,
    inputs: NPArray,
    outputs: NPArray,
    test_cases: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    config = ModelStressPlotterConfigTreloar()
    num_model_inputs = 256
    num_model_samples = 16

    figure_all, axes_all = plt.subplots()

    def plot_one_input_output_set(
        inputs: NPArray, test_case_identifier: int, outputs: NPArray
    ) -> None:
        if test_case_identifier == test_case_identifier_uniaxial_tension:
            test_case_label = "uniaxial_tension"
            data_marker = config.data_marker_ut
            data_color = config.data_color_ut
            data_label = config.data_label_ut
            model_color_mean = config.model_color_ut
            model_color_credible_interval = config.model_color_ut
            model_label_mean = config.model_label_mean_ut
            model_label_samples = config.model_label_samples_ut
        elif test_case_identifier == test_case_identifier_equibiaxial_tension:
            test_case_label = "equibiaxial_tension"
            data_marker = config.data_marker_ebt
            data_color = config.data_color_ebt
            data_label = config.data_label_ebt
            model_color_mean = config.model_color_ebt
            model_color_credible_interval = config.model_color_ebt
            model_label_mean = config.model_label_mean_ebt
            model_label_samples = config.model_label_samples_ebt
        else:
            test_case_label = "pure_shear"
            data_marker = config.data_marker_ps
            data_color = config.data_color_ps
            data_label = config.data_label_ps
            model_color_mean = config.model_color_ps
            model_color_credible_interval = config.model_color_ps
            model_label_mean = config.model_label_mean_ps
            model_label_samples = config.model_label_samples_ps

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
        model_test_cases = np.full((num_model_inputs,), test_case_identifier)
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
            linewidth=config.model_mean_linewidth,
            label=model_label_mean,
        )
        axes.fill_between(
            model_stretches_plot,
            min_quantiles_plot,
            max_quantiles_plot,
            color=model_color_credible_interval,
            alpha=config.model_credible_interval_alpha,
        )
        axes_all.plot(
            model_stretches_plot,
            means_plot,
            color=model_color_mean,
            linewidth=config.model_mean_linewidth,
            label=model_label_mean,
        )
        axes_all.fill_between(
            model_stretches_plot,
            min_quantiles_plot,
            max_quantiles_plot,
            color=model_color_credible_interval,
            alpha=config.model_credible_interval_alpha,
        )

        samples = sample_from_model(
            model,
            parameter_samples[:num_model_samples, :],
            model_stretches,
            model_test_cases,
            device,
        )
        for sample_counter, sample in enumerate(samples):
            sample_plot = sample.reshape((-1,))
            if sample_counter == (num_model_samples - 1):
                axes.plot(
                    model_stretches_plot,
                    sample_plot,
                    color=config.model_samples_color,
                    linewidth=config.model_samples_linewidth,
                    alpha=config.model_samples_alpha,
                    label=model_label_samples,
                )
            axes.plot(
                model_stretches_plot,
                sample,
                color=config.model_samples_color,
                linewidth=config.model_samples_linewidth,
                alpha=config.model_samples_alpha,
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
        metrics_test_cases = np.full((num_data_inputs,), test_case_identifier)
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
            0.72,
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
            0.03,
            0.5,
            text,
            transform=axes_all.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

        output_path = project_directory.create_output_file_path(
            file_name=file_name, subdir_name=output_subdirectory
        )
        figure_all.savefig(output_path, bbox_inches="tight", dpi=config.dpi)

    input_sets, test_case_identifiers, output_sets = split_treloar_inputs_and_outputs(
        inputs, test_cases, outputs
    )

    for input_set, test_case_identifier, output_set in zip(
        input_sets, test_case_identifiers, output_sets
    ):
        plot_one_input_output_set(input_set, test_case_identifier, output_set)

    plot_all_input_and_output_sets()
    plt.clf()


class ModelStressPlotterConfigKawabata:
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
        # credible interval
        self.model_credible_interval_alpha = 0.2

        # legend
        self.legend_color = "gray"

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_model_stresses_kawabata(
    model: IsotropicModelLibrary,
    parameter_samples: NPArray,
    inputs: NPArray,
    outputs: NPArray,
    test_cases: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    config = ModelStressPlotterConfigKawabata()
    set_sizes = [1, 6, 6, 8, 8, 8, 8, 8, 7, 7, 6, 3]
    num_sets = len(set_sizes)
    num_model_inputs_per_set = 128

    stretches_2 = inputs[:, 1]
    min_stretch_2 = np.amin(stretches_2)
    max_stretch_2 = np.amax(stretches_2)

    def plot_one_stress_dimension(
        input_sets: list[NPArray],
        test_case_identifiers: list[int],
        output_sets: list[NPArray],
        stress_dim: int,
    ) -> None:
        file_name = f"kawabata_data_stress_{stress_dim}.png"

        figure, axes = plt.subplots()
        color_map = plt.get_cmap(config.color_map, num_sets)
        colors = color_map(np.linspace(0, 1, num_sets))

        for set_index, input_set, test_case_identifier, output_set in zip(
            range(num_sets), input_sets, test_case_identifiers, output_sets
        ):
            data_stretches = input_set
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
            model_test_cases = np.full(
                (num_model_inputs_per_set,), test_case_identifier
            )

            means, _ = calculate_model_mean_and_stddev(
                model,
                parameter_samples,
                model_stretches,
                model_test_cases,
                device,
                output_dim=stress_dim,
            )
            min_quantiles, max_quantiles = calculate_model_quantiles(
                model,
                parameter_samples,
                model_stretches,
                model_test_cases,
                device,
                output_dim=stress_dim,
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
                alpha=config.model_credible_interval_alpha,
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
        model_credible_interval = Patch(
            facecolor=config.legend_color,
            alpha=config.model_credible_interval_alpha,
            label="95%-credible interval",
        )
        data_legend_handles, _ = axes.get_legend_handles_labels()
        legend_handles = [
            data_point,
            model_mean,
            model_credible_interval,
        ] + data_legend_handles
        axes.legend(
            handles=legend_handles,
            fontsize=config.font_size,
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )

        # text box metrics
        num_data_inputs = len(inputs)
        metrics_test_cases = np.full((num_data_inputs,), test_case_identifier)
        coverage = calclulate_coverage(
            model,
            parameter_samples,
            inputs,
            metrics_test_cases,
            outputs,
            device,
            output_dim=stress_dim,
        )
        r_squared = calculate_coefficient_of_determinant(
            model,
            parameter_samples,
            inputs,
            metrics_test_cases,
            outputs,
            device,
            output_dim=stress_dim,
        )
        rmse = calculate_root_mean_squared_error(
            model,
            parameter_samples,
            inputs,
            metrics_test_cases,
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

    input_sets, test_case_identifiers, output_sets = split_kawabata_inputs_and_outputs(
        inputs, test_cases, outputs
    )
    plot_one_stress_dimension(
        input_sets, test_case_identifiers, output_sets, stress_dim=0
    )
    plot_one_stress_dimension(
        input_sets, test_case_identifiers, output_sets, stress_dim=1
    )
    plt.clf()


class ModelStressPlotterConfigLinka:
    def __init__(self) -> None:
        # label size
        self.label_size = 10
        # font size in legend
        self.font_size = 12
        self.font: Dict[str, Any] = {"size": self.font_size}

        ## ticks
        self.num_x_tick_labels = 6
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
        self.data_label = "data"
        self.data_marker = "x"
        self.data_color = "tab:blue"
        self.data_marker_size = 5
        # model
        self.model_label = "model"
        # mean
        self.model_color_mean = "tab:blue"
        # credible interval
        self.model_credible_interval_alpha = 0.2
        self.model_color_credible_interval = "tab:blue"

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_model_stresses_linka(
    model: OrthotropicCANN,
    parameter_samples: NPArray,
    inputs: NPArray,
    outputs: NPArray,
    test_cases: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    config = ModelStressPlotterConfigLinka()
    num_data_sets_simple_shear = 6
    num_data_sets_biaxial_tension = 5
    num_data_sets = num_data_sets_simple_shear + num_data_sets_biaxial_tension
    num_points_per_data_set = 11
    min_principal_stretch = 1.0
    max_principal_stretch = 1.1
    min_shear_strain = 0.0
    max_shear_strain = 0.5
    input_labels = [
        r"$\gamma_{sf}$" + " [-]",
        r"$\gamma_{nf}$" + " [-]",
        r"$\gamma_{fs}$" + " [-]",
        r"$\gamma_{ns}$" + " [-]",
        r"$\gamma_{fn}$" + " [-]",
        r"$\gamma_{sn}$" + " [-]",
        r"$\lambda$" + " [-]",
        r"$\lambda$" + " [-]",
        r"$\lambda$" + " [-]",
        r"$\lambda$" + " [-]",
        r"$\lambda$" + " [-]",
    ]
    stress_labels = [
        r"$\sigma_{ff}$" + " [kPa]",
        r"$\sigma_{fs}$" + " [kPa]",
        r"$\sigma_{fn}$" + " [kPa]",
        r"$\sigma_{sf}$" + " [kPa]",
        r"$\sigma_{ss}$" + " [kPa]",
        r"$\sigma_{sn}$" + " [kPa]",
        r"$\sigma_{nf}$" + " [kPa]",
        r"$\sigma_{ns}$" + " [kPa]",
        r"$\sigma_{nn}$" + " [kPa]",
    ]
    stress_file_name_labels = [
        "fiber",
        "fs",
        "fn",
        "sf",
        "sheet",
        "sn",
        "nf",
        "ns",
        "normal",
    ]
    stretch_ratios = [
        (1.0, 1.0),
        (1.0, 0.75),
        (0.75, 1.0),
        (1.0, 0.5),
        (0.5, 1.0),
    ]
    stretch_ratio_index_fiber = 0
    stretch_ratio_index_normal = 1
    index_principal_stress_f = 0
    index_shear_stress_fs = 1
    index_shear_stress_fn = 2
    index_shear_stress_sf = 3
    index_shear_stress_sn = 4
    index_shear_stress_nf = 5
    index_shear_stress_ns = 6
    index_principal_stress_n = 7
    principal_stress_indices = [index_principal_stress_f, index_principal_stress_n]
    shear_stress_indices_plots = [
        [index_shear_stress_fs],
        [index_shear_stress_fn],
        [index_shear_stress_sf],
        [index_shear_stress_sn],
        [index_shear_stress_nf],
        [index_shear_stress_ns],
    ]
    principal_stress_indices_plots = [
        principal_stress_indices for _ in range(num_data_sets_biaxial_tension)
    ]
    stress_indices_list = shear_stress_indices_plots + principal_stress_indices_plots
    num_model_inputs = 256

    def plot_one_data_set(
        inputs: NPArray,
        test_case_identifier: int,
        outputs: NPArray,
        stress_indices: list[int],
        data_set_index: int,
    ) -> None:

        def plot_one_stress(stress_index: int) -> None:
            is_principal_stress = stress_index in principal_stress_indices
            stress_file_name_label = stress_file_name_labels[stress_index]

            figure, axes = plt.subplots()

            if is_principal_stress:
                min_input = min_principal_stretch
                max_input = max_principal_stretch
                principal_stretch_data_set_index = (
                    data_set_index - num_data_sets_simple_shear
                )
                stretch_ratio = stretch_ratios[principal_stretch_data_set_index]
                stretch_ratio_fiber = stretch_ratio[stretch_ratio_index_fiber]
                stretch_ratio_normal = stretch_ratio[stretch_ratio_index_normal]
                file_name = f"principalstress_{stress_file_name_label}_stretchratio_{stretch_ratio_fiber}_{stretch_ratio_normal}.pdf"
            else:
                min_input = min_shear_strain
                max_input = max_shear_strain
                file_name = f"shearstress_{stress_file_name_label}.pdf"

            # data points
            data_inputs_axis = np.linspace(
                min_input, max_input, num_points_per_data_set
            )
            data_stresses = outputs[:, stress_index]
            axes.plot(
                data_inputs_axis,
                data_stresses,
                marker=config.data_marker,
                color=config.data_color,
                markersize=config.data_marker_size,
                linestyle="None",
                label=config.data_label,
            )

            # model
            model_inputs_axis = np.linspace(min_input, max_input, num_model_inputs)
            min_model_inputs = np.min(inputs, axis=0)
            max_model_inputs = np.max(inputs, axis=0)
            model_inputs = np.linspace(
                min_model_inputs, max_model_inputs, num_model_inputs
            )
            model_test_cases = np.full((num_model_inputs,), test_case_identifier)

            means, _ = calculate_model_mean_and_stddev(
                model,
                parameter_samples,
                model_inputs,
                model_test_cases,
                device,
                output_dim=stress_index,
            )
            min_quantiles, max_quantiles = calculate_model_quantiles(
                model,
                parameter_samples,
                model_inputs,
                model_test_cases,
                device,
                output_dim=stress_index,
            )
            means = means.reshape((-1,))
            min_quantiles = min_quantiles.reshape((-1,))
            max_quantiles = max_quantiles.reshape((-1,))

            axes.plot(
                model_inputs_axis,
                means,
                color=config.model_color_mean,
                label=config.model_label,
            )
            axes.fill_between(
                model_inputs_axis,
                min_quantiles,
                max_quantiles,
                color=config.model_color_credible_interval,
                alpha=config.model_credible_interval_alpha,
            )

            # axis ticks
            x_ticks = np.linspace(
                min_input,
                max_input,
                num=config.num_x_tick_labels,
            )
            x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
            axes.set_xticks(x_ticks)
            axes.set_xticklabels(x_tick_labels)

            # axis labels
            input_label = input_labels[data_set_index]
            axes.set_xlabel(input_label, **config.font)
            stress_label = stress_labels[stress_index]
            axes.set_ylabel(stress_label, **config.font)
            axes.tick_params(
                axis="both", which="minor", labelsize=config.minor_tick_label_size
            )
            axes.tick_params(
                axis="both", which="major", labelsize=config.major_tick_label_size
            )

            # legend
            model_credible_interval = Patch(
                facecolor=config.model_color_credible_interval,
                alpha=config.model_credible_interval_alpha,
                label="95%-credible interval",
            )
            data_legend_handles, _ = axes.get_legend_handles_labels()
            legend_handles = data_legend_handles + [model_credible_interval]
            axes.legend(
                handles=legend_handles,
                fontsize=config.font_size,
                loc="upper left",
            )

            # text box ratios
            if is_principal_stress:
                text = "\n".join(
                    (
                        r"$\lambda_{f}=%.2f \, \lambda$" % (stretch_ratio_fiber,),
                        r"$\lambda_{n}=%.2f \, \lambda$" % (stretch_ratio_normal,),
                    )
                )
                text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
                axes.text(
                    0.55,
                    0.96,
                    text,
                    transform=axes.transAxes,
                    fontsize=config.font_size,
                    verticalalignment="top",
                    bbox=text_properties,
                )

            # text box metrics
            num_data_inputs = len(inputs)
            metrics_test_cases = np.full((num_data_inputs,), test_case_identifier)
            coverage = calclulate_coverage(
                model,
                parameter_samples,
                inputs,
                metrics_test_cases,
                outputs,
                device,
                output_dim=stress_index,
            )
            r_squared = calculate_coefficient_of_determinant(
                model,
                parameter_samples,
                inputs,
                metrics_test_cases,
                outputs,
                device,
                output_dim=stress_index,
            )
            rmse = calculate_root_mean_squared_error(
                model,
                parameter_samples,
                inputs,
                metrics_test_cases,
                outputs,
                device,
                output_dim=stress_index,
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
                0.7,
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

        for stress_index in stress_indices:
            plot_one_stress(stress_index)

    input_sets, test_case_identifiers, output_sets = split_linka_inputs_and_outputs(
        inputs, test_cases, outputs
    )

    for (
        input_set,
        test_case_identifier,
        output_set,
        stress_indices,
        data_set_index,
    ) in zip(
        input_sets,
        test_case_identifiers,
        output_sets,
        stress_indices_list,
        range(num_data_sets),
    ):
        plot_one_data_set(
            input_set, test_case_identifier, output_set, stress_indices, data_set_index
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


def sample_from_model(
    model: ModelProtocol,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> NPArray:
    num_samples = len(parameter_samples)
    samples = calculate_model_predictions(
        model=model,
        parameter_samples=parameter_samples,
        inputs=inputs,
        test_cases=test_cases,
        device=device,
    )

    if output_dim is not None:
        samples = samples[:, :, output_dim].reshape((num_samples, -1, 1))

    return samples


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


################################################################################
# Gaussian processes
################################################################################


class GPStressPlotterConfigTreloar:
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
        # gp
        self.gp_color_ut = "tab:blue"
        self.gp_color_ebt = "tab:purple"
        self.gp_color_ps = "tab:green"
        # mean
        self.gp_label_mean_ut = "mean ut"
        self.gp_label_mean_ebt = "mean ebt"
        self.gp_label_mean_ps = "mean ps"
        self.gp_mean_linewidth = 1.0
        # credible interval
        self.gp_credible_interval_alpha = 0.4
        # samples
        self.gp_label_samples_ut = "samples ut"
        self.gp_label_samples_ebt = "samples ebt"
        self.gp_label_samples_ps = "samples ps"
        self.gp_samples_color = "tab:gray"
        self.gp_samples_linewidth = 1.0
        self.gp_samples_alpha = 0.2

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_gp_stresses_treloar(
    gaussian_process: GaussianProcess,
    inputs: NPArray,
    outputs: NPArray,
    test_cases: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    config = GPStressPlotterConfigTreloar()
    num_gp_samples = 16
    num_gp_inputs = 256

    figure_all, axes_all = plt.subplots()

    def plot_one_input_output_set(
        inputs: NPArray, test_case: int, outputs: NPArray
    ) -> None:
        if test_case == test_case_identifier_uniaxial_tension:
            test_case_label = "uniaxial_tension"
            data_marker = config.data_marker_ut
            data_color = config.data_color_ut
            data_label = config.data_label_ut
            gp_color = config.gp_color_ut
            gp_color_credible_interval = config.gp_color_ut
            gp_label_mean = config.gp_label_mean_ut
            gp_label_samples = config.gp_label_samples_ut
        elif test_case == test_case_identifier_equibiaxial_tension:
            test_case_label = "equibiaxial_tension"
            data_marker = config.data_marker_ebt
            data_color = config.data_color_ebt
            data_label = config.data_label_ebt
            gp_color = config.gp_color_ebt
            gp_color_credible_interval = config.gp_color_ebt
            gp_label_mean = config.gp_label_mean_ebt
            gp_label_samples = config.gp_label_samples_ebt
        else:
            test_case_label = "pure_shear"
            data_marker = config.data_marker_ps
            data_color = config.data_color_ps
            data_label = config.data_label_ps
            gp_color = config.gp_color_ps
            gp_color_credible_interval = config.gp_color_ps
            gp_label_mean = config.gp_label_mean_ps
            gp_label_samples = config.gp_label_samples_ps

        file_name = f"treloar_data_{test_case_label}.png"

        figure, axes = plt.subplots()

        stretches = inputs[:, 0]
        min_stretch = np.amin(stretches)
        max_stretch = np.amax(stretches)

        # data points
        data_stretches = stretches
        data_stresses = outputs
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

        # gp
        gp_stretches_plot = np.linspace(min_stretch, max_stretch, num_gp_inputs)

        gp_stretches = (
            torch.from_numpy(gp_stretches_plot.reshape((-1, 1)))
            .type(torch.get_default_dtype())
            .to(device)
        )
        gp_test_cases = torch.full((num_gp_inputs,), test_case, device=device)
        gp_inputs = assemble_stretches_from_factors(gp_stretches, gp_test_cases, device)

        means = calculate_gp_means(gaussian_process, gp_inputs)
        min_quantiles, max_quantiles = calculate_gp_quantiles(
            gaussian_process, gp_inputs
        )

        means_plot = means.reshape((-1,))
        min_quantiles_plot = min_quantiles.reshape((-1,))
        max_quantiles_plot = max_quantiles.reshape((-1,))

        axes.plot(
            gp_stretches_plot,
            means_plot,
            color=gp_color,
            linewidth=config.gp_mean_linewidth,
            label=gp_label_mean,
        )
        axes.fill_between(
            gp_stretches_plot,
            min_quantiles_plot,
            max_quantiles_plot,
            color=gp_color_credible_interval,
            alpha=config.gp_credible_interval_alpha,
        )
        axes_all.plot(
            gp_stretches_plot,
            means_plot,
            color=gp_color,
            linewidth=config.gp_mean_linewidth,
            label=gp_label_mean,
        )
        axes_all.fill_between(
            gp_stretches_plot,
            min_quantiles_plot,
            max_quantiles_plot,
            color=gp_color_credible_interval,
            alpha=config.gp_credible_interval_alpha,
        )

        samples = sample_from_gp(gaussian_process, gp_inputs, num_gp_samples)
        for sample_counter, sample in enumerate(samples):
            if sample_counter == (num_gp_samples - 1):
                axes.plot(
                    gp_stretches_plot,
                    sample,
                    color=config.gp_samples_color,
                    linewidth=config.gp_samples_linewidth,
                    alpha=config.gp_samples_alpha,
                    label=gp_label_samples,
                )
            axes.plot(
                gp_stretches_plot,
                sample,
                color=config.gp_samples_color,
                linewidth=config.gp_samples_linewidth,
                alpha=config.gp_samples_alpha,
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

    input_sets, test_case_sets, output_sets = split_treloar_inputs_and_outputs(
        inputs, test_cases, outputs
    )

    for input_set, test_case, output_set in zip(
        input_sets, test_case_sets, output_sets
    ):
        plot_one_input_output_set(input_set, test_case, output_set)

    plot_all_input_and_output_sets()
    plt.clf()


def calculate_gp_means(
    gaussian_process: GaussianProcess,
    inputs: Tensor,
    output_dim: Optional[int] = None,
) -> NPArray:
    gp = reduce_gp_to_output_dimension(gaussian_process, output_dim)
    predictive_distribution = infer_predictive_distribution(gp, inputs)
    _means = predictive_distribution.mean
    means = _means.cpu().detach().numpy()
    return means


def calculate_gp_quantiles(
    gaussian_process: GaussianProcess,
    inputs: Tensor,
    output_dim: Optional[int] = None,
) -> tuple[NPArray, NPArray]:
    gp = reduce_gp_to_output_dimension(gaussian_process, output_dim)
    predictive_distribution = infer_predictive_distribution(gp, inputs)
    means = predictive_distribution.mean
    stddevs = predictive_distribution.stddev
    _min_quantiles = means - factor_stddev_credible_interval * stddevs
    _max_quantiles = means + factor_stddev_credible_interval * stddevs
    min_quantiles = _min_quantiles.cpu().detach().numpy()
    max_quantiles = _max_quantiles.cpu().detach().numpy()
    return min_quantiles, max_quantiles


def sample_from_gp(
    gaussian_process: GaussianProcess,
    inputs: Tensor,
    num_samples: int,
    output_dim: Optional[int] = None,
) -> NPArray:
    gp = reduce_gp_to_output_dimension(gaussian_process, output_dim)
    predictive_distribution = infer_predictive_distribution(gp, inputs)
    _samples = predictive_distribution.sample(sample_shape=torch.Size((num_samples,)))
    samples = _samples.cpu().detach().numpy()
    return samples


def infer_predictive_distribution(
    gaussian_process: GaussianProcess, inputs: Tensor
) -> GPMultivariateNormal:
    likelihood = gaussian_process.likelihood
    return likelihood(gaussian_process(inputs))


def reduce_gp_to_output_dimension(
    gaussian_process: GaussianProcess, output_dim: int | None
) -> GP:
    _validate_gp_and_output_dimension(gaussian_process, output_dim)
    is_multi_output_gp = isinstance(gaussian_process, IndependentMultiOutputGP)
    is_output_dim_defined = not (output_dim == None)

    if is_multi_output_gp and is_output_dim_defined:
        return gaussian_process.get_gp_for_one_output_dimension(cast(int, output_dim))
    else:
        return gaussian_process


def _validate_gp_and_output_dimension(
    gaussian_process: GaussianProcess, output_dim: int | None
) -> None:
    is_multi_output_gp = isinstance(gaussian_process, IndependentMultiOutputGP)
    is_output_dim_defined = not (output_dim == None)
    if is_multi_output_gp and not is_output_dim_defined:
        raise PlotterError(
            """For independent multi-output GPs, 
            the output dimension must be defined for the evaluation."""
        )
    elif not is_multi_output_gp and is_output_dim_defined:
        raise PlotterError("No output dimension can be defined for single-output GPs")
