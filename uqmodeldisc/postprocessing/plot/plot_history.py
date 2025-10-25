import math
from typing import Any, Dict

import matplotlib.pyplot as plt

from uqmodeldisc.io import ProjectDirectory


class HistoryPlotterConfig:
    def __init__(self) -> None:
        # label size
        self.label_size = 16
        # font size in legend
        self.font_size = 16
        self.font: Dict[str, Any] = {"size": self.font_size}

        # major ticks
        self.major_tick_label_size = 20
        self.major_ticks_size = self.font_size
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 14
        self.minor_ticks_size = 12
        self.minor_ticks_width = 1

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_loss_history(
    loss_hists: list[list[float]],
    loss_hist_names: list[str],
    file_name: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: HistoryPlotterConfig,
) -> None:
    figure, axes = plt.subplots()
    axes.set_title("Loss history", **config.font)
    for loss_hist, loss_hist_name in zip(loss_hists, loss_hist_names):
        axes.plot(loss_hist, label=loss_hist_name)
    axes.set_yscale("log")
    axes.set_ylabel("MSE", **config.font)
    axes.set_xlabel("epoch", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    axes.legend(fontsize=config.font_size, loc="best")
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()


def plot_statistical_loss_history(
    loss_hist: list[float],
    statistical_quantity: str,
    file_name: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: HistoryPlotterConfig,
    max_limit: float = 1e32,
) -> None:
    def find_y_limits() -> tuple[float, float]:
        min_value = min(loss_hist)
        max_value = max(loss_hist)
        floored_order_min_value = int(math.floor(math.log10(abs(min_value))))
        floored_order_max_value = int(math.floor(math.log10(abs(max_value))))
        if min_value > 0:
            y_min = 10 ** (floored_order_min_value)
        else:
            y_min = -(10 ** (floored_order_min_value + 1))

        if max_value > 0:
            y_max = 10 ** (floored_order_max_value + 1)
        else:
            y_max = -(10 ** (floored_order_max_value))
        y_max = min(max_limit, y_max)
        return float(y_min), float(y_max)

    figure, axes = plt.subplots()
    axes.plot(loss_hist)
    axes.set_ylabel(statistical_quantity, **config.font)
    y_min, y_max = find_y_limits()
    axes.set_yscale("symlog")
    axes.set_ylim(ymin=y_min, ymax=y_max)
    axes.set_xlabel("iteration", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()
