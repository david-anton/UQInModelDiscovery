from dataclasses import dataclass
from typing import Protocol, TypeAlias

import gpytorch
import normflows as nf
import numpy.typing as npt
import pandas as pd
import torch
from matplotlib import axes, figure

# Pytorch
Tensor: TypeAlias = torch.Tensor
TensorSize: TypeAlias = torch.Size
Parameter: TypeAlias = torch.nn.Parameter
ParameterList: TypeAlias = torch.nn.ParameterList
Module: TypeAlias = torch.nn.Module
Device: TypeAlias = torch.device
TorchOptimizer: TypeAlias = torch.optim.Optimizer
TorchLRScheduler: TypeAlias = torch.optim.lr_scheduler.LRScheduler

# GPyTorch
GPModel: TypeAlias = gpytorch.models.ExactGP
GPMean: TypeAlias = gpytorch.means.Mean
GPKernel: TypeAlias = gpytorch.kernels.Kernel
GPLikelihood: TypeAlias = gpytorch.likelihoods.Likelihood

# normflows
NFFlow: TypeAlias = nf.flows.Flow
NFNormalizingFlow: TypeAlias = nf.NormalizingFlow
NFBaseDistribution: TypeAlias = nf.distributions.BaseDistribution

# Numpy
NPArray: TypeAlias = npt.NDArray

# Pandas
PDDataFrame: TypeAlias = pd.DataFrame
PDIndex: TypeAlias = pd.Index

# Matplotlib
PLTFigure: TypeAlias = figure.Figure
PLTAxes: TypeAlias = axes.Axes


# dataclass
@dataclass
class DataClass(Protocol):
    pass
