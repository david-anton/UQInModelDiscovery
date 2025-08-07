from typing import Any, Dict, Optional, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from torch import vmap

from bayesianmdisc.customtypes import Device, NPArray, Tensor
from bayesianmdisc.data import interpolate_heteroscedastic_noise
from bayesianmdisc.data.linkaheartdataset import (
    assemble_flattened_deformation_gradients,
    generate_principal_stretches,
)
from bayesianmdisc.errors import PlotterError
from bayesianmdisc.gps.base import GPMultivariateNormal
from bayesianmdisc.gps.gp import GP
from bayesianmdisc.gps.multioutputgp import IndependentMultiOutputGP
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import IsotropicModel, ModelProtocol, OrthotropicCANN
from bayesianmdisc.models.base_mechanics import assemble_stretches_from_factors
from bayesianmdisc.postprocessing.plot.utility import (
    split_kawabata_inputs_and_outputs,
    split_linka_inputs_and_outputs,
    split_linka_noise_stddevs,
    split_treloar_inputs_and_outputs,
    split_treloar_noise_stddevs,
)
from bayesianmdisc.statistics.metrics import (
    coefficient_of_determination,
    gp_coverage_test,
    model_coverage_test,
    root_mean_squared_error,
)
from bayesianmdisc.statistics.utility import determine_quantiles_from_samples
from bayesianmdisc.testcases import (
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.utility import from_numpy_to_torch, from_torch_to_numpy
from bayesianmdisc.datasettings import create_four_terms_linka_model_parameters

GaussianProcess: TypeAlias = GP | IndependentMultiOutputGP
MetricList: TypeAlias = list[float]
OutputList: TypeAlias = list[NPArray]

credible_interval = 0.95
factor_stddev_credible_interval = 1.96
cm_to_inch = 1 / 2.54

################################################################################
# Model
################################################################################


class ModelStressPlotterConfigTreloar:
    def __init__(self) -> None:
        # label size
        self.label_size = 7
        # font size in legend
        self.font_size = 7
        self.font: Dict[str, Any] = {"size": self.font_size}
        # figure size
        self.figure_size = (16 * cm_to_inch, 12 * cm_to_inch)
        self.pad_subplots_width = 0.8
        self.pad_subplots_hight = 1.2

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

        ### stresses
        # data
        self.data_label_ut = "data UT"
        self.data_label_ebt = "data EBT"
        self.data_label_ps = "data PS"
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
        self.model_mean_label_ut = "mean UT"
        self.model_mean_label_ebt = "mean EBT"
        self.model_mean_label_ps = "mean PS"
        self.model_mean_linewidth = 1.0
        # credible interval
        self.model_credible_interval_alpha = 0.4
        # samples
        self.model_samples_label_ut = "samples UT"
        self.model_samples_label_ebt = "samples EBT"
        self.model_samples_label_ps = "samples PS"
        self.model_samples_color = "tab:gray"
        self.model_samples_linewidth = 1.0
        self.model_samples_alpha = 0.2

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_model_stresses_treloar(
    model: IsotropicModel,
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

    file_name = f"model.png"
    figure, axes = plt.subplots(2, 2, figsize=config.figure_size)
    figure.tight_layout(
        w_pad=config.pad_subplots_width, h_pad=config.pad_subplots_hight
    )
    axis_all = axes[1, 1]

    def plot_one_input_output_set(
        inputs: NPArray, test_case_identifier: int, outputs: NPArray
    ) -> None:
        if test_case_identifier == test_case_identifier_uniaxial_tension:
            axis = axes[0, 0]
            data_marker = config.data_marker_ut
            data_color = config.data_color_ut
            data_label = config.data_label_ut
            model_color_mean = config.model_color_ut
            model_color_credible_interval = config.model_color_ut
            model_label_mean = config.model_mean_label_ut
            model_label_samples = config.model_samples_label_ut
        elif test_case_identifier == test_case_identifier_equibiaxial_tension:
            axis = axes[0, 1]
            data_marker = config.data_marker_ebt
            data_color = config.data_color_ebt
            data_label = config.data_label_ebt
            model_color_mean = config.model_color_ebt
            model_color_credible_interval = config.model_color_ebt
            model_label_mean = config.model_mean_label_ebt
            model_label_samples = config.model_samples_label_ebt
        else:
            axis = axes[1, 0]
            data_marker = config.data_marker_ps
            data_color = config.data_color_ps
            data_label = config.data_label_ps
            model_color_mean = config.model_color_ps
            model_color_credible_interval = config.model_color_ps
            model_label_mean = config.model_mean_label_ps
            model_label_samples = config.model_samples_label_ps

        data_stretches = inputs[:, 0]
        data_stresses = outputs
        min_stretch = np.amin(data_stretches)
        max_stretch = np.amax(data_stretches)

        # data points
        axis.plot(
            data_stretches,
            data_stresses,
            marker=data_marker,
            color=data_color,
            markersize=config.data_marker_size,
            linestyle="None",
            label=data_label,
        )
        axis_all.plot(
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
        mean_model_stresses, _ = calculate_model_mean_and_stddev(
            model,
            parameter_samples,
            model_stretches,
            model_test_cases,
            device,
        )
        min_quantile_model_stresses, max_quantile_model_stresses = (
            calculate_model_quantiles(
                model, parameter_samples, model_stretches, model_test_cases, device
            )
        )

        model_stretches_plot = model_stretches.reshape((-1,))
        means_plot = mean_model_stresses.reshape((-1,))
        min_quantiles_plot = min_quantile_model_stresses.reshape((-1,))
        max_quantiles_plot = max_quantile_model_stresses.reshape((-1,))

        axis.plot(
            model_stretches_plot,
            means_plot,
            color=model_color_mean,
            linewidth=config.model_mean_linewidth,
            label=model_label_mean,
        )
        axis.fill_between(
            model_stretches_plot,
            min_quantiles_plot,
            max_quantiles_plot,
            color=model_color_credible_interval,
            alpha=config.model_credible_interval_alpha,
        )
        axis_all.plot(
            model_stretches_plot,
            means_plot,
            color=model_color_mean,
            linewidth=config.model_mean_linewidth,
            label=model_label_mean,
        )
        axis_all.fill_between(
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
                axis.plot(
                    model_stretches_plot,
                    sample_plot,
                    color=config.model_samples_color,
                    linewidth=config.model_samples_linewidth,
                    alpha=config.model_samples_alpha,
                    label=model_label_samples,
                )
            axis.plot(
                model_stretches_plot,
                sample,
                color=config.model_samples_color,
                linewidth=config.model_samples_linewidth,
                alpha=config.model_samples_alpha,
            )

        # axis ticks
        x_ticks = np.linspace(min_stretch, max_stretch, num=config.num_x_ticks)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(x_tick_labels)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=config.num_y_ticks))

        # axis labels
        axis.set_xlabel("stretch [-]", **config.font)
        axis.set_ylabel("stress [kPa]", **config.font)
        axis.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axis.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )

        # legend
        model_credible_interval = Patch(
            facecolor=model_color_credible_interval,
            alpha=config.model_credible_interval_alpha,
            label="95%-credible interval",
        )
        data_legend_handles, _ = axis.get_legend_handles_labels()
        legend_handles = data_legend_handles + [
            model_credible_interval,
        ]
        axis.legend(handles=legend_handles, fontsize=config.font_size, loc="upper left")

        # text box metrics
        num_data_inputs = len(inputs)
        metrics_test_cases = np.full((num_data_inputs,), test_case_identifier)
        coverage = calclulate_model_coverage(
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
        axis.text(
            0.68,
            0.18,
            text,
            transform=axis.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

    def plot_all_input_and_output_sets() -> None:
        # data
        data_stretches = inputs[:, 0]
        min_stretch = np.amin(data_stretches)
        max_stretch = np.amax(data_stretches)

        # axis ticks
        x_ticks = np.linspace(min_stretch, max_stretch, num=config.num_x_ticks)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axis_all.set_xticks(x_ticks)
        axis_all.set_xticklabels(x_tick_labels)

        # axis labels
        axis_all.set_xlabel("stretch [-]", **config.font)
        axis_all.set_ylabel("stress [kPa]", **config.font)
        axis_all.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axis_all.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )
        axis_all.yaxis.set_major_locator(MaxNLocator(nbins=config.num_y_ticks))

        # # legend
        # axis_all.legend(fontsize=config.font_size, loc="upper left")

        # text box metrics
        coverage = calclulate_model_coverage(
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
                "Total " + r"$C_{95\%}=$" + r"${0}\%$".format(round(coverage, 2)),
                "Total " + r"$R^{2}=$" + r"${0}$".format(round(r_squared, 4)),
                "Total " + r"$RMSE=$" + r"${0}$".format(round(rmse, 4)),
            )
        )
        text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
        axis_all.text(
            0.03,
            0.96,
            text,
            transform=axis_all.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

    input_sets, test_case_identifiers, output_sets = split_treloar_inputs_and_outputs(
        inputs, test_cases, outputs
    )

    for input_set, test_case_identifier, output_set in zip(
        input_sets, test_case_identifiers, output_sets
    ):
        plot_one_input_output_set(input_set, test_case_identifier, output_set)

    plot_all_input_and_output_sets()
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
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
        self.model_credible_interval_alpha = 0.4

        # legend
        self.legend_color = "gray"

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_model_stresses_kawabata(
    model: IsotropicModel,
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

            mean_model_stresses, _ = calculate_model_mean_and_stddev(
                model,
                parameter_samples,
                model_stretches,
                model_test_cases,
                device,
                output_dim=stress_dim,
            )
            min_quantile_model_stresses, max_quantile_model_stresses = (
                calculate_model_quantiles(
                    model,
                    parameter_samples,
                    model_stretches,
                    model_test_cases,
                    device,
                    output_dim=stress_dim,
                )
            )
            model_stretches_plot = model_stretches_2.reshape((-1,))
            means_plot = mean_model_stresses.reshape((-1,))
            min_quantiles_plot = min_quantile_model_stresses.reshape((-1,))
            max_quantiles_plot = max_quantile_model_stresses.reshape((-1,))

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
        coverage = calclulate_model_coverage(
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


class LinkaDataConfig:
    def __init__(self) -> None:
        self.num_data_sets_simple_shear = 6
        self.num_data_sets_biaxial_tension = 5
        self.num_data_sets = (
            self.num_data_sets_simple_shear + self.num_data_sets_biaxial_tension
        )
        self.min_principal_stretch = 1.0
        self.max_principal_stretch = 1.1
        self.min_shear_strain = 0.0
        self.max_shear_strain = 0.5
        self.input_labels = [
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
        self.stress_labels = [
            r"$\sigma_{ff}$" + " [kPa]",
            r"$\sigma_{fs}$" + " [kPa]",
            r"$\sigma_{fn}$" + " [kPa]",
            r"$\sigma_{sf}$" + " [kPa]",
            r"$\sigma_{sn}$" + " [kPa]",
            r"$\sigma_{nf}$" + " [kPa]",
            r"$\sigma_{ns}$" + " [kPa]",
            r"$\sigma_{nn}$" + " [kPa]",
        ]
        self.stress_file_name_labels = [
            "fiber",
            "fs",
            "fn",
            "sf",
            "sn",
            "nf",
            "ns",
            "normal",
        ]
        self.stretch_ratios = [
            (1.0, 1.0),
            (1.0, 0.75),
            (0.75, 1.0),
            (1.0, 0.5),
            (0.5, 1.0),
        ]
        self.stretch_ratio_index_fiber = 0
        self.stretch_ratio_index_normal = 1
        self.index_principal_stress_f = 0
        self.index_shear_stress_fs = 1
        self.index_shear_stress_fn = 2
        self.index_shear_stress_sf = 3
        self.index_shear_stress_sn = 4
        self.index_shear_stress_nf = 5
        self.index_shear_stress_ns = 6
        self.index_principal_stress_n = 7
        self.principal_stress_indices = [
            self.index_principal_stress_f,
            self.index_principal_stress_n,
        ]
        self.shear_stress_indices_plots = [
            [self.index_shear_stress_fs],
            [self.index_shear_stress_fn],
            [self.index_shear_stress_sf],
            [self.index_shear_stress_sn],
            [self.index_shear_stress_nf],
            [self.index_shear_stress_ns],
        ]
        self.principal_stress_indices_plots = [
            self.principal_stress_indices
            for _ in range(self.num_data_sets_biaxial_tension)
        ]
        self.stress_indices_list = (
            self.shear_stress_indices_plots + self.principal_stress_indices_plots
        )
        self.subfigure_indices = [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
            [3, 0],
            [3, 1],
            [3, 2],
            [4, 0],
            [4, 1],
            [4, 2],
            [5, 0],
        ]


class ModelStressPlotterConfigLinka:
    def __init__(self) -> None:
        # label size
        self.label_size = 7
        # font size in legend
        self.font_size = 7
        self.ratio_font_size = 5
        self.font: Dict[str, Any] = {"size": self.font_size}

        # figure size
        self.figure_size = (16 * cm_to_inch, 24 * cm_to_inch)
        self.pad_subplots = 0.8

        ## ticks
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

        ### stresses
        # data
        self.data_label = "data"
        self.data_marker = "x"
        self.data_color = "tab:blue"
        self.data_marker_size = 5
        # model
        self.model_label = "mean"
        self.model_color = "tab:blue"
        # four term model
        self.four_term_model_label = "Four-term-model (Martonová et al., 2024)"
        self.four_term_model_color = "tab:red"
        self.four_term_model_linestyle = "dashed"
        self.four_term_model_linewidth = 1.0
        self.four_term_model_alpha = 0.8
        # credible interval
        self.model_credible_interval_alpha = 0.4
        # samples
        self.model_samples_label = "samples"
        self.model_samples_color = "tab:gray"
        self.model_samples_linewidth = 1.0
        self.model_samples_alpha = 0.2

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_model_stresses_linka(
    model: OrthotropicCANN,
    parameter_samples: NPArray,
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    num_points_per_test_case: int,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
    plot_four_term_model: bool = False,
) -> None:
    plotter_config = ModelStressPlotterConfigLinka()
    data_config = LinkaDataConfig()
    num_model_inputs = 256
    num_model_samples = 16

    file_name = f"model.png"
    figure, axes = plt.subplots(6, 3, figsize=plotter_config.figure_size)
    figure.tight_layout(pad=plotter_config.pad_subplots)
    axes[5, 1].axis("off")
    axes[5, 2].axis("off")
    subfigure_counter = 0
    coverage_list: MetricList = []
    data_output_list: OutputList = []
    mean_model_output_list: OutputList = []

    def plot_one_data_set(
        inputs: NPArray,
        test_case_identifier: int,
        outputs: NPArray,
        stress_indices: list[int],
        data_set_index: int,
        subfigure_counter: int,
        coverage_list: MetricList,
        data_output_list: OutputList,
        mean_model_output_list: OutputList,
    ) -> tuple[int, MetricList, OutputList, OutputList]:

        def plot_one_stress(
            stress_index: int,
            subfigure_counter: int,
            coverage_list: MetricList,
            data_output_list: OutputList,
            mean_model_output_list: OutputList,
        ) -> tuple[int, MetricList, OutputList, OutputList]:
            is_principal_stress = stress_index in data_config.principal_stress_indices

            subfigure_indices = data_config.subfigure_indices[subfigure_counter]
            axis = axes[subfigure_indices[0], subfigure_indices[1]]

            if is_principal_stress:
                principal_stretch_data_set_index = (
                    data_set_index - data_config.num_data_sets_simple_shear
                )
                stretch_ratio = data_config.stretch_ratios[
                    principal_stretch_data_set_index
                ]
                stretch_ratio_fiber = stretch_ratio[
                    data_config.stretch_ratio_index_fiber
                ]
                stretch_ratio_normal = stretch_ratio[
                    data_config.stretch_ratio_index_normal
                ]
                _model_inputs = generate_principal_stretches(
                    stretch_ratio, num_model_inputs
                )
                # model inputs
                min_input_axis = data_config.min_principal_stretch
                # if stress_index == data_config.index_principal_stress_f:
                #     _input_index = data_config.stretch_ratio_index_fiber
                # elif stress_index == data_config.index_principal_stress_n:
                #     _input_index = data_config.stretch_ratio_index_normal
                # max_input = np.amax(_model_inputs[:, _input_index])
                max_input_axis = data_config.max_principal_stretch
                model_inputs_axis = np.linspace(
                    min_input_axis, max_input_axis, num_model_inputs
                )
            else:
                # model inputs
                min_input_axis = data_config.min_shear_strain
                max_input_axis = data_config.max_shear_strain
                model_inputs_axis = np.linspace(
                    min_input_axis, max_input_axis, num_model_inputs
                )
                _model_inputs = model_inputs_axis.reshape((-1, 1))

            # data points
            data_inputs_axis = np.linspace(
                min_input_axis, max_input_axis, num_points_per_test_case
            )
            data_stresses = outputs[:, stress_index]
            axis.plot(
                data_inputs_axis,
                data_stresses,
                marker=plotter_config.data_marker,
                color=plotter_config.data_color,
                markersize=plotter_config.data_marker_size,
                linestyle="None",
                label=plotter_config.data_label,
            )

            # model
            model_inputs = assemble_flattened_deformation_gradients(
                _model_inputs, test_case_identifier
            )
            model_test_cases = np.full((num_model_inputs,), test_case_identifier)

            mean_model_stresses, _ = calculate_model_mean_and_stddev(
                model,
                parameter_samples,
                model_inputs,
                model_test_cases,
                device,
                output_dim=stress_index,
            )
            min_quantile_model_stresses, max_quantile_model_stresses = (
                calculate_model_quantiles(
                    model,
                    parameter_samples,
                    model_inputs,
                    model_test_cases,
                    device,
                    output_dim=stress_index,
                )
            )
            mean_model_stresses = mean_model_stresses.reshape((-1,))
            min_quantile_model_stresses = min_quantile_model_stresses.reshape((-1,))
            max_quantile_model_stresses = max_quantile_model_stresses.reshape((-1,))

            axis.plot(
                model_inputs_axis,
                mean_model_stresses,
                color=plotter_config.model_color,
                label=plotter_config.model_label,
            )
            axis.fill_between(
                model_inputs_axis,
                min_quantile_model_stresses,
                max_quantile_model_stresses,
                color=plotter_config.model_color,
                alpha=plotter_config.model_credible_interval_alpha,
            )

            samples = sample_from_model(
                model,
                parameter_samples[:num_model_samples, :],
                model_inputs,
                model_test_cases,
                device,
                output_dim=stress_index,
            )
            for sample_counter, sample in enumerate(samples):
                sample_plot = sample.reshape((-1,))
                if sample_counter == (num_model_samples - 1):
                    axis.plot(
                        model_inputs_axis,
                        sample_plot,
                        color=plotter_config.model_samples_color,
                        linewidth=plotter_config.model_samples_linewidth,
                        alpha=plotter_config.model_samples_alpha,
                        label=plotter_config.model_samples_label,
                    )
                axis.plot(
                    model_inputs_axis,
                    sample,
                    color=plotter_config.model_samples_color,
                    linewidth=plotter_config.model_samples_linewidth,
                    alpha=plotter_config.model_samples_alpha,
                )

            # four term model
            if plot_four_term_model:
                four_term_model, four_term_model_parameters = (
                    create_four_terms_linka_model(device)
                )
                four_term_model_output = calculate_model_predictions(
                    four_term_model,
                    four_term_model_parameters,
                    model_inputs,
                    model_test_cases,
                    device,
                )[0, :, stress_index].reshape(-1, 1)
                axis.plot(
                    model_inputs_axis,
                    four_term_model_output,
                    color=plotter_config.four_term_model_color,
                    label=plotter_config.four_term_model_label,
                    linestyle=plotter_config.four_term_model_linestyle,
                    alpha=plotter_config.four_term_model_alpha,
                    linewidth=plotter_config.four_term_model_linewidth,
                )

            # axis ticks
            x_ticks = np.linspace(
                min_input_axis,
                max_input_axis,
                num=plotter_config.num_x_ticks,
            )
            x_tick_labels = [str(round(tick, 3)) for tick in x_ticks]
            axis.set_xticks(x_ticks)
            axis.set_xticklabels(x_tick_labels)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=plotter_config.num_y_ticks))

            # axis labels
            input_label = data_config.input_labels[data_set_index]
            axis.set_xlabel(input_label, **plotter_config.font)
            stress_label = data_config.stress_labels[stress_index]
            axis.set_ylabel(stress_label, **plotter_config.font)
            axis.tick_params(
                axis="both",
                which="minor",
                labelsize=plotter_config.minor_tick_label_size,
            )
            axis.tick_params(
                axis="both",
                which="major",
                labelsize=plotter_config.major_tick_label_size,
            )

            # text box ratios
            if is_principal_stress:
                text = "\n".join(
                    (
                        r"$\lambda_{f}=1+%.2f(\lambda-1)$" % (stretch_ratio_fiber,),
                        r"$\lambda_{n}=1+%.2f(\lambda-1)$" % (stretch_ratio_normal,),
                    )
                )
                text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
                axis.text(
                    0.04,
                    0.60,
                    text,
                    transform=axis.transAxes,
                    fontsize=plotter_config.ratio_font_size,
                    verticalalignment="top",
                    bbox=text_properties,
                )

            # text box metrics
            num_data_inputs = len(inputs)
            metrics_test_cases = np.full((num_data_inputs,), test_case_identifier)
            coverage = calclulate_model_coverage(
                model,
                parameter_samples,
                inputs,
                metrics_test_cases,
                outputs,
                device,
                output_dim=stress_index,
            )
            coverage_list += [coverage]
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
            axis.text(
                0.04,
                0.95,
                text,
                transform=axis.transAxes,
                fontsize=plotter_config.font_size,
                verticalalignment="top",
                bbox=text_properties,
            )

            # model outputs for total metrics
            data_outputs = outputs[:, stress_index].reshape((-1, 1))
            data_output_list += [data_outputs]
            mean_model_output, _ = calculate_model_mean_and_stddev(
                model,
                parameter_samples,
                inputs,
                metrics_test_cases,
                device,
                output_dim=stress_index,
            )
            mean_model_output_list += [mean_model_output]

            if subfigure_counter == 15:
                # legend
                model_credible_interval = Patch(
                    facecolor=plotter_config.model_color,
                    alpha=plotter_config.model_credible_interval_alpha,
                    label="95%-credible interval",
                )
                # data_legend_handles, _ = axis.get_legend_handles_labels()
                # legend_handles = data_legend_handles.insert(3, model_credible_interval)
                # axis.legend(
                #     handles=legend_handles,
                #     fontsize=plotter_config.font_size,
                #     bbox_to_anchor=(1.15, 0.95),
                #     loc="upper left",
                #     borderaxespad=0.0,
                # )
                legend_handles, _ = axis.get_legend_handles_labels()
                legend_handles.insert(3, model_credible_interval)
                axis.legend(
                    handles=legend_handles,
                    fontsize=plotter_config.font_size,
                    bbox_to_anchor=(1.15, 0.95),
                    loc="upper left",
                    borderaxespad=0.0,
                )

                total_coverage = np.mean(np.array(coverage_list))
                total_data_outputs = np.concatenate(data_output_list, axis=0)
                total_mean_model_outputs = np.concatenate(
                    mean_model_output_list, axis=0
                )
                total_r_squared = coefficient_of_determination(
                    total_mean_model_outputs, total_data_outputs
                )
                total_rmse = root_mean_squared_error(
                    total_mean_model_outputs, total_data_outputs
                )
                text = "\n".join(
                    (
                        "Total "
                        + r"$C_{95\%}=$"
                        + r"${0}\%$".format(round(total_coverage, 2)),
                        "Total "
                        + r"$R^{2}=$"
                        + r"${0}$".format(round(total_r_squared, 4)),
                        "Total " + r"$RMSE=$" + r"${0}$".format(round(total_rmse, 4)),
                    )
                )
                text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
                if plot_four_term_model:
                    x_position_text = 2.8
                else:
                    x_position_text = 2.18

                axis.text(
                    x_position_text,
                    0.92,
                    text,
                    transform=axis.transAxes,
                    fontsize=plotter_config.font_size,
                    verticalalignment="top",
                    bbox=text_properties,
                )

            subfigure_counter += 1
            return (
                subfigure_counter,
                coverage_list,
                data_output_list,
                mean_model_output_list,
            )

        for stress_index in stress_indices:
            (
                subfigure_counter,
                coverage_list,
                data_output_list,
                mean_model_output_list,
            ) = plot_one_stress(
                stress_index,
                subfigure_counter,
                coverage_list,
                data_output_list,
                mean_model_output_list,
            )

        return (
            subfigure_counter,
            coverage_list,
            data_output_list,
            mean_model_output_list,
        )

    input_sets, test_case_identifiers, output_sets = split_linka_inputs_and_outputs(
        inputs, test_cases, outputs, num_points_per_test_case
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
        data_config.stress_indices_list,
        range(data_config.num_data_sets),
    ):
        subfigure_counter, coverage_list, data_output_list, mean_model_output_list = (
            plot_one_data_set(
                input_set,
                test_case_identifier,
                output_set,
                stress_indices,
                data_set_index,
                subfigure_counter,
                coverage_list,
                data_output_list,
                mean_model_output_list,
            )
        )

    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=plotter_config.dpi)
    plt.clf()


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
    min_quantiles, max_quantiles = determine_quantiles_from_samples(
        prediction_samples, credible_interval
    )

    if output_dim is not None:
        min_quantiles = min_quantiles[:, output_dim].reshape((-1, 1))
        max_quantiles = max_quantiles[:, output_dim].reshape((-1, 1))

    return min_quantiles, max_quantiles


def calclulate_model_coverage(
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

    return model_coverage_test(prediction_samples, outputs, credible_interval)


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


class GPStressPlotterConfigTreloar(ModelStressPlotterConfigTreloar):
    def __init__(self) -> None:
        super().__init__()


def plot_gp_stresses_treloar(
    gaussian_process: GaussianProcess,
    inputs: NPArray,
    outputs: NPArray,
    test_cases: NPArray,
    noise_stddevs: NPArray,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    config = GPStressPlotterConfigTreloar()
    num_gp_samples = 8
    num_gp_inputs = 256

    file_name = f"gaussian_processes.png"
    figure, axes = plt.subplots(2, 2, figsize=config.figure_size)
    figure.tight_layout(
        w_pad=config.pad_subplots_width, h_pad=config.pad_subplots_hight
    )
    axis_all = axes[1, 1]

    def plot_one_input_output_set(
        input_set: NPArray,
        test_case_identifier: int,
        output_set: NPArray,
        noise_stddev_set: NPArray,
    ) -> None:
        if test_case_identifier == test_case_identifier_uniaxial_tension:
            axis = axes[0, 0]
            data_marker = config.data_marker_ut
            data_color = config.data_color_ut
            data_label = config.data_label_ut
            gp_color = config.model_color_ut
            gp_color_credible_interval = config.model_color_ut
            gp_label_mean = config.model_mean_label_ut
            gp_label_samples = config.model_samples_label_ut
        elif test_case_identifier == test_case_identifier_equibiaxial_tension:
            axis = axes[0, 1]
            data_marker = config.data_marker_ebt
            data_color = config.data_color_ebt
            data_label = config.data_label_ebt
            gp_color = config.model_color_ebt
            gp_color_credible_interval = config.model_color_ebt
            gp_label_mean = config.model_mean_label_ebt
            gp_label_samples = config.model_samples_label_ebt
        else:
            axis = axes[1, 0]
            data_marker = config.data_marker_ps
            data_color = config.data_color_ps
            data_label = config.data_label_ps
            gp_color = config.model_color_ps
            gp_color_credible_interval = config.model_color_ps
            gp_label_mean = config.model_mean_label_ps
            gp_label_samples = config.model_samples_label_ps

        test_case_set = np.full((len(input_set),), test_case_identifier, dtype=np.int64)

        stretches = input_set[:, 0]
        min_stretch = np.amin(stretches)
        max_stretch = np.amax(stretches)

        # data points
        data_stretches = stretches
        data_stresses = output_set
        axis.plot(
            data_stretches,
            data_stresses,
            marker=data_marker,
            color=data_color,
            markersize=config.data_marker_size,
            linestyle="None",
            label=data_label,
        )
        axis_all.plot(
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

        gp_stretches = from_numpy_to_torch(gp_stretches_plot.reshape((-1, 1)), device)
        gp_test_cases_torch = torch.full(
            (num_gp_inputs,), test_case_identifier, device=device
        )
        gp_inputs_torch = assemble_stretches_from_factors(
            gp_stretches, gp_test_cases_torch, device
        )
        gp_inputs = from_torch_to_numpy(gp_inputs_torch)

        gp_noise_stddevs_torch = interpolate_heteroscedastic_noise(
            new_inputs=gp_inputs_torch,
            new_test_cases=gp_test_cases_torch,
            inputs=from_numpy_to_torch(input_set, device),
            test_cases=from_numpy_to_torch(test_case_set, device),
            noise_stddevs=from_numpy_to_torch(noise_stddev_set, device),
            device=device,
        )
        gp_noise_stddevs = from_torch_to_numpy(gp_noise_stddevs_torch)

        mean_gp_stresses = calculate_gp_means(
            gaussian_process, gp_inputs, gp_noise_stddevs, device
        )
        min_quantile_gp_stresses, max_quantile_gp_stresses = calculate_gp_quantiles(
            gaussian_process, gp_inputs, gp_noise_stddevs, device
        )

        means_plot = mean_gp_stresses.reshape((-1,))
        min_quantiles_plot = min_quantile_gp_stresses.reshape((-1,))
        max_quantiles_plot = max_quantile_gp_stresses.reshape((-1,))

        axis.plot(
            gp_stretches_plot,
            means_plot,
            color=gp_color,
            linewidth=config.model_mean_linewidth,
            label=gp_label_mean,
        )
        axis.fill_between(
            gp_stretches_plot,
            min_quantiles_plot,
            max_quantiles_plot,
            color=gp_color_credible_interval,
            alpha=config.model_credible_interval_alpha,
        )
        axis_all.plot(
            gp_stretches_plot,
            means_plot,
            color=gp_color,
            linewidth=config.model_mean_linewidth,
            label=gp_label_mean,
        )
        axis_all.fill_between(
            gp_stretches_plot,
            min_quantiles_plot,
            max_quantiles_plot,
            color=gp_color_credible_interval,
            alpha=config.model_credible_interval_alpha,
        )

        samples = sample_from_gp(
            gaussian_process, gp_inputs, gp_noise_stddevs, num_gp_samples, device
        )
        for sample_counter, sample in enumerate(samples):
            if sample_counter == (num_gp_samples - 1):
                axis.plot(
                    gp_stretches_plot,
                    sample,
                    color=config.model_samples_color,
                    linewidth=config.model_samples_linewidth,
                    alpha=config.model_samples_alpha,
                    label=gp_label_samples,
                )
            axis.plot(
                gp_stretches_plot,
                sample,
                color=config.model_samples_color,
                linewidth=config.model_samples_linewidth,
                alpha=config.model_samples_alpha,
            )

        # axis ticks
        x_ticks = np.linspace(min_stretch, max_stretch, num=config.num_x_ticks)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(x_tick_labels)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=config.num_y_ticks))

        # axis labels
        axis.set_xlabel("stretch [-]", **config.font)
        axis.set_ylabel("stress [kPa]", **config.font)
        axis.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axis.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )

        # legend
        model_credible_interval = Patch(
            facecolor=gp_color_credible_interval,
            alpha=config.model_credible_interval_alpha,
            label="95%-credible interval",
        )
        data_legend_handles, _ = axis.get_legend_handles_labels()
        legend_handles = data_legend_handles + [
            model_credible_interval,
        ]
        axis.legend(handles=legend_handles, fontsize=config.font_size, loc="upper left")

        # text box metrics
        coverage = calclulate_gp_coverage(
            gaussian_process, input_set, output_set, noise_stddev_set, device
        )
        text = "\n".join((r"$C_{95\%}=$" + r"${0}\%$".format(round(coverage, 2)),))
        text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
        axis.text(
            0.69,
            0.08,
            text,
            transform=axis.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

    def plot_all_input_and_output_sets() -> None:
        # data
        data_stretches = inputs[:, 0]
        min_stretch = np.amin(data_stretches)
        max_stretch = np.amax(data_stretches)

        # axis ticks
        x_ticks = np.linspace(min_stretch, max_stretch, num=config.num_x_ticks)
        x_tick_labels = [str(round(tick, 2)) for tick in x_ticks]
        axis_all.set_xticks(x_ticks)
        axis_all.set_xticklabels(x_tick_labels)

        # axis labels
        axis_all.set_xlabel("stretch [-]", **config.font)
        axis_all.set_ylabel("stress [kPa]", **config.font)
        axis_all.tick_params(
            axis="both", which="minor", labelsize=config.minor_tick_label_size
        )
        axis_all.tick_params(
            axis="both", which="major", labelsize=config.major_tick_label_size
        )
        axis_all.yaxis.set_major_locator(MaxNLocator(nbins=config.num_y_ticks))

        # # legend
        # axes_all.legend(fontsize=config.font_size, loc="upper left")

        # text box metrics
        coverage = calclulate_gp_coverage(
            gaussian_process,
            inputs,
            outputs,
            noise_stddevs,
            device,
        )
        text = "\n".join(
            ("Total " + r"$C_{95\%}=$" + r"${0}\%$".format(round(coverage, 2)),)
        )
        text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
        axis_all.text(
            0.03,
            0.96,
            text,
            transform=axis_all.transAxes,
            fontsize=config.font_size,
            verticalalignment="top",
            bbox=text_properties,
        )

    input_sets, test_case_sets, output_sets = split_treloar_inputs_and_outputs(
        inputs, test_cases, outputs
    )
    noise_stddev_sets = split_treloar_noise_stddevs(noise_stddevs, test_cases)

    for input_set, test_case, output_set, noise_stddev_set in zip(
        input_sets, test_case_sets, output_sets, noise_stddev_sets
    ):
        plot_one_input_output_set(input_set, test_case, output_set, noise_stddev_set)

    plot_all_input_and_output_sets()
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()


class GPStressPlotterConfigLinka:
    def __init__(self) -> None:
        # label size
        self.label_size = 7
        # font size in legend
        self.font_size = 7
        self.ratio_font_size = 5
        self.font: Dict[str, Any] = {"size": self.font_size}
        # figure size
        self.figure_size = (16 * cm_to_inch, 24 * cm_to_inch)
        self.pad_subplots = 0.8

        ## ticks
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

        ### stresses
        # data
        self.data_label = "data"
        self.data_marker = "x"
        self.data_color = "tab:blue"
        self.data_marker_size = 5
        ### gp
        self.gp_color = "tab:blue"
        # mean
        self.gp_mean_label = "mean"
        # credible interval
        self.gp_credible_interval_alpha = 0.4
        # samples
        self.gp_samples_label = "samples"
        self.gp_samples_color = "tab:gray"
        self.gp_samples_linewidth = 1.0
        self.gp_samples_alpha = 0.2

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_gp_stresses_linka(
    gaussian_process: GaussianProcess,
    inputs: NPArray,
    test_cases: NPArray,
    outputs: NPArray,
    noise_stddevs: NPArray,
    num_points_per_test_case: int,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    plotter_config = GPStressPlotterConfigLinka()
    data_config = LinkaDataConfig()
    num_gp_samples = 8
    num_gp_inputs = 256

    file_name = f"gaussian_processes.png"
    figure, axes = plt.subplots(6, 3, figsize=plotter_config.figure_size)
    figure.tight_layout(pad=plotter_config.pad_subplots)
    axes[5, 1].axis("off")
    axes[5, 2].axis("off")
    subfigure_counter = 0
    coverage_list: MetricList = []

    def plot_one_data_set(
        input_set: NPArray,
        test_case_identifier: int,
        output_set: NPArray,
        noise_stddev_set: NPArray,
        stress_indices: list[int],
        data_set_index: int,
        subfigure_counter: int,
        coverage_list: MetricList,
    ) -> tuple[int, MetricList]:

        def plot_one_stress(
            stress_index: int, subfigure_counter: int, coverage_list: MetricList
        ) -> tuple[int, MetricList]:
            is_principal_stress = stress_index in data_config.principal_stress_indices
            test_case_set = np.full(
                (len(input_set),), test_case_identifier, dtype=np.int64
            )

            subfigure_indices = data_config.subfigure_indices[subfigure_counter]
            axis = axes[subfigure_indices[0], subfigure_indices[1]]

            if is_principal_stress:
                principal_stretch_data_set_index = (
                    data_set_index - data_config.num_data_sets_simple_shear
                )
                stretch_ratio = data_config.stretch_ratios[
                    principal_stretch_data_set_index
                ]
                stretch_ratio_fiber = stretch_ratio[
                    data_config.stretch_ratio_index_fiber
                ]
                stretch_ratio_normal = stretch_ratio[
                    data_config.stretch_ratio_index_normal
                ]
                _gp_inputs = generate_principal_stretches(stretch_ratio, num_gp_inputs)
                # model inputs
                min_input_axis = data_config.min_principal_stretch
                # if stress_index == data_config.index_principal_stress_f:
                #     _input_index = data_config.stretch_ratio_index_fiber
                # elif stress_index == data_config.index_principal_stress_n:
                #     _input_index = data_config.stretch_ratio_index_normal
                # max_input = np.amax(_gp_inputs[:, _input_index])
                max_input_axis = data_config.max_principal_stretch
                gp_inputs_axis = np.linspace(
                    min_input_axis, max_input_axis, num_gp_inputs
                )
            else:
                # model inputs
                min_input_axis = data_config.min_shear_strain
                max_input_axis = data_config.max_shear_strain
                gp_inputs_axis = np.linspace(
                    min_input_axis, max_input_axis, num_gp_inputs
                )
                _gp_inputs = gp_inputs_axis.reshape((-1, 1))

            # data points
            data_inputs_axis = np.linspace(
                min_input_axis, max_input_axis, num_points_per_test_case
            )
            data_stresses = output_set[:, stress_index]
            axis.plot(
                data_inputs_axis,
                data_stresses,
                marker=plotter_config.data_marker,
                color=plotter_config.data_color,
                markersize=plotter_config.data_marker_size,
                linestyle="None",
                label=plotter_config.data_label,
            )

            # GP
            gp_inputs = assemble_flattened_deformation_gradients(
                _gp_inputs, test_case_identifier
            )
            gp_inputs_torch = from_numpy_to_torch(gp_inputs, device)
            gp_test_cases_torch = torch.full(
                (num_gp_inputs,), test_case_identifier, device=device
            )
            gp_noise_stddevs_torch = interpolate_heteroscedastic_noise(
                new_inputs=gp_inputs_torch,
                new_test_cases=gp_test_cases_torch,
                inputs=from_numpy_to_torch(input_set, device),
                test_cases=from_numpy_to_torch(test_case_set, device),
                noise_stddevs=from_numpy_to_torch(noise_stddev_set, device),
                device=device,
            )
            gp_noise_stddevs = from_torch_to_numpy(gp_noise_stddevs_torch)

            mean_gp_stresses = calculate_gp_means(
                gaussian_process,
                gp_inputs,
                gp_noise_stddevs,
                device,
                output_dim=stress_index,
            )
            min_quantile_gp_stresses, max_quantile_gp_stresses = calculate_gp_quantiles(
                gaussian_process,
                gp_inputs,
                gp_noise_stddevs,
                device,
                output_dim=stress_index,
            )

            means_plot = mean_gp_stresses.reshape((-1,))
            min_quantiles_plot = min_quantile_gp_stresses.reshape((-1,))
            max_quantiles_plot = max_quantile_gp_stresses.reshape((-1,))

            axis.plot(
                gp_inputs_axis,
                means_plot,
                color=plotter_config.gp_color,
                label=plotter_config.gp_mean_label,
            )
            axis.fill_between(
                gp_inputs_axis,
                min_quantiles_plot,
                max_quantiles_plot,
                color=plotter_config.gp_color,
                alpha=plotter_config.gp_credible_interval_alpha,
            )

            samples = sample_from_gp(
                gaussian_process,
                gp_inputs,
                gp_noise_stddevs,
                num_gp_samples,
                device,
                output_dim=stress_index,
            )
            for sample_counter, sample in enumerate(samples):
                if sample_counter == (num_gp_samples - 1):
                    axis.plot(
                        gp_inputs_axis,
                        sample,
                        color=plotter_config.gp_samples_color,
                        linewidth=plotter_config.gp_samples_linewidth,
                        alpha=plotter_config.gp_samples_alpha,
                        label=plotter_config.gp_samples_label,
                    )
                axis.plot(
                    gp_inputs_axis,
                    sample,
                    color=plotter_config.gp_samples_color,
                    linewidth=plotter_config.gp_samples_linewidth,
                    alpha=plotter_config.gp_samples_alpha,
                )

            # axis ticks
            x_ticks = np.linspace(
                min_input_axis,
                max_input_axis,
                num=plotter_config.num_x_ticks,
            )
            x_tick_labels = [str(round(tick, 3)) for tick in x_ticks]
            axis.set_xticks(x_ticks)
            axis.set_xticklabels(x_tick_labels)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=plotter_config.num_y_ticks))

            # axis labels
            input_label = data_config.input_labels[data_set_index]
            axis.set_xlabel(input_label, **plotter_config.font)
            stress_label = data_config.stress_labels[stress_index]
            axis.set_ylabel(stress_label, **plotter_config.font)
            axis.tick_params(
                axis="both",
                which="minor",
                labelsize=plotter_config.minor_tick_label_size,
            )
            axis.tick_params(
                axis="both",
                which="major",
                labelsize=plotter_config.major_tick_label_size,
            )

            # text box ratios
            if is_principal_stress:
                text = "\n".join(
                    (
                        r"$\lambda_{f}=1+%.2f(\lambda-1)$" % (stretch_ratio_fiber,),
                        r"$\lambda_{n}=1+%.2f(\lambda-1)$" % (stretch_ratio_normal,),
                    )
                )
                text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
                axis.text(
                    0.04,
                    0.76,
                    text,
                    transform=axis.transAxes,
                    fontsize=plotter_config.ratio_font_size,
                    verticalalignment="top",
                    bbox=text_properties,
                )

            # text box metrics
            coverage = calclulate_gp_coverage(
                gaussian_process,
                input_set,
                output_set,
                noise_stddev_set,
                device,
                output_dim=stress_index,
            )
            coverage_list += [coverage]
            text = "\n".join((r"$C_{95\%}=$" + r"${0}\%$".format(round(coverage, 2)),))
            text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
            axis.text(
                0.04,
                0.95,
                text,
                transform=axis.transAxes,
                fontsize=plotter_config.font_size,
                verticalalignment="top",
                bbox=text_properties,
            )

            # legend + overall coverage
            if subfigure_counter == 15:
                model_credible_interval = Patch(
                    facecolor=plotter_config.gp_color,
                    alpha=plotter_config.gp_credible_interval_alpha,
                    label="95%-credible interval",
                )
                data_legend_handles, _ = axis.get_legend_handles_labels()
                legend_handles = data_legend_handles + [model_credible_interval]
                axis.legend(
                    handles=legend_handles,
                    fontsize=plotter_config.font_size,
                    bbox_to_anchor=(1.15, 0.95),
                    loc="upper left",
                    borderaxespad=0.0,
                )

                total_coverage = np.mean(np.array(coverage_list))
                text = "\n".join(
                    (
                        "Total "
                        + r"$C_{95\%}=$"
                        + r"${0}\%$".format(round(total_coverage, 2)),
                    )
                )
                text_properties = dict(boxstyle="square", facecolor="white", alpha=1.0)
                axis.text(
                    2.18,
                    0.92,
                    text,
                    transform=axis.transAxes,
                    fontsize=plotter_config.font_size,
                    verticalalignment="top",
                    bbox=text_properties,
                )

            subfigure_counter += 1
            return subfigure_counter, coverage_list

        for stress_index in stress_indices:
            subfigure_counter, coverage_list = plot_one_stress(
                stress_index, subfigure_counter, coverage_list
            )

        return subfigure_counter, coverage_list

    input_sets, test_case_identifiers, output_sets = split_linka_inputs_and_outputs(
        inputs, test_cases, outputs, num_points_per_test_case
    )
    noise_stddev_sets = split_linka_noise_stddevs(
        noise_stddevs, num_points_per_test_case
    )

    for (
        input_set,
        test_case_identifier,
        output_set,
        noise_stddev_set,
        stress_indices,
        data_set_index,
    ) in zip(
        input_sets,
        test_case_identifiers,
        output_sets,
        noise_stddev_sets,
        data_config.stress_indices_list,
        range(data_config.num_data_sets),
    ):
        subfigure_counter, coverage_list = plot_one_data_set(
            input_set,
            test_case_identifier,
            output_set,
            noise_stddev_set,
            stress_indices,
            data_set_index,
            subfigure_counter,
            coverage_list,
        )

    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=plotter_config.dpi)
    plt.clf()


def calculate_gp_means(
    gaussian_process: GaussianProcess,
    inputs: NPArray,
    noise_stddevs: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> NPArray:
    predictive_distribution = infer_predictive_distribution(
        gaussian_process=gaussian_process,
        inputs=inputs,
        noise_stddevs=noise_stddevs,
        device=device,
        output_dim=output_dim,
    )
    means_torch = predictive_distribution.mean
    return from_torch_to_numpy(means_torch)


def calculate_gp_quantiles(
    gaussian_process: GaussianProcess,
    inputs: NPArray,
    noise_stddevs: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> tuple[NPArray, NPArray]:
    predictive_distribution = infer_predictive_distribution(
        gaussian_process=gaussian_process,
        inputs=inputs,
        noise_stddevs=noise_stddevs,
        device=device,
        output_dim=output_dim,
    )
    means_torch = predictive_distribution.mean
    stddevs_torch = predictive_distribution.stddev
    min_quantiles_torch = means_torch - factor_stddev_credible_interval * stddevs_torch
    max_quantiles_torch = means_torch + factor_stddev_credible_interval * stddevs_torch
    min_quantiles = from_torch_to_numpy(min_quantiles_torch)
    max_quantiles = from_torch_to_numpy(max_quantiles_torch)
    return min_quantiles, max_quantiles


def sample_from_gp(
    gaussian_process: GaussianProcess,
    inputs: NPArray,
    noise_stddevs: NPArray,
    num_samples: int,
    device: Device,
    output_dim: Optional[int] = None,
) -> NPArray:
    predictive_distribution = infer_predictive_distribution(
        gaussian_process=gaussian_process,
        inputs=inputs,
        noise_stddevs=noise_stddevs,
        device=device,
        output_dim=output_dim,
    )
    samples_torch = predictive_distribution.sample(
        sample_shape=torch.Size((num_samples,))
    )
    return from_torch_to_numpy(samples_torch)


def infer_predictive_distribution(
    gaussian_process: GaussianProcess,
    inputs: NPArray,
    noise_stddevs: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> GPMultivariateNormal:
    inputs_torch = from_numpy_to_torch(inputs, device)
    noise_stddevs = reduce_noise_stddevs_to_output_dim(noise_stddevs, output_dim)
    noise_stddevs_torch = from_numpy_to_torch(noise_stddevs, device)
    gp = reduce_gp_to_output_dimension(gaussian_process, output_dim)
    return infer_predictive_distribution_for_one_dimension(
        gp, inputs_torch, noise_stddevs_torch
    )


def infer_predictive_distribution_for_one_dimension(
    gaussian_process: GaussianProcess, inputs: Tensor, noise_stddevs: Tensor
) -> GPMultivariateNormal:
    # return gaussian_process.infer_predictive_distribution(inputs, noise_stddevs)
    return gaussian_process(inputs)


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


def reduce_noise_stddevs_to_output_dim(
    noise_stddevs: NPArray, output_dim: int | None
) -> NPArray:
    _validate_noise_standard_deviations_and_output_dimension(noise_stddevs, output_dim)
    is_multi_output_noise_stddevs = not (noise_stddevs.shape[1] == 1)
    is_output_dim_defined = not (output_dim == None)

    if is_multi_output_noise_stddevs and is_output_dim_defined:
        return noise_stddevs[:, output_dim].reshape((-1, 1))
    else:
        return noise_stddevs


def calclulate_gp_coverage(
    gaussian_process: GaussianProcess,
    inputs: NPArray,
    outputs: NPArray,
    noise_stddevs: NPArray,
    device: Device,
    output_dim: Optional[int] = None,
) -> float:
    min_quantiles, max_quantiles = calculate_gp_quantiles(
        gaussian_process=gaussian_process,
        inputs=inputs,
        noise_stddevs=noise_stddevs,
        device=device,
        output_dim=output_dim,
    )

    if output_dim is not None:
        outputs = outputs[:, output_dim]
    else:
        outputs = outputs.reshape((-1,))

    return gp_coverage_test(min_quantiles, max_quantiles, outputs)


def _validate_gp_and_output_dimension(
    gaussian_process: GaussianProcess, output_dim: int | None
) -> None:
    is_multi_output_gp = isinstance(gaussian_process, IndependentMultiOutputGP)
    is_output_dim_defined = not (output_dim == None)
    if is_multi_output_gp and not is_output_dim_defined:
        raise PlotterError(
            """For independent multi-output GPs, 
            the output dimension must be defined for evaluation."""
        )
    elif not is_multi_output_gp and is_output_dim_defined:
        raise PlotterError("Output dimension can not be defined for single-output GPs")


def _validate_noise_standard_deviations_and_output_dimension(
    noise_stddevs: NPArray, output_dim: int | None
) -> None:
    is_multi_output_noise_stddevs = not (noise_stddevs.shape[1] == 1)
    is_output_dim_defined = not (output_dim == None)
    if is_multi_output_noise_stddevs and not is_output_dim_defined:
        raise PlotterError(
            """For multi-output noise standard deviations, 
            the output dimension must be defined for evaluation."""
        )
    elif not is_multi_output_noise_stddevs and is_output_dim_defined:
        raise PlotterError(
            """Output dimension can not be defined for 
            single-output noise standard deviations"""
        )


def create_four_terms_linka_model(device: Device) -> tuple[OrthotropicCANN, NPArray]:
    model = OrthotropicCANN(device)
    four_terms_model_parameters = create_four_terms_linka_model_parameters()
    parameter_names = four_terms_model_parameters.names
    parameter_values = np.array(four_terms_model_parameters.values).reshape(1, -1)
    model.reduce_model_to_parameter_names(parameter_names)
    return model, parameter_values
