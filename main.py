import os
from dataclasses import dataclass
from datetime import date
from typing import cast

import torch

from bayesianmdisc.bayes.likelihood import Likelihood
from bayesianmdisc.bayes.prior import (
    PriorProtocol,
    create_independent_multivariate_gamma_distributed_prior,
)
from bayesianmdisc.customtypes import Device, NPArray, Tensor
from bayesianmdisc.data import (
    DataReaderProtocol,
    DeformationInputs,
    KawabataDataReader,
    LinkaHeartDataReader,
    StressOutputs,
    TestCases,
    TreloarDataReader,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_uniaxial_tension,
)
from bayesianmdisc.errors import CombinedPriorError, DataError, DataSetError
from bayesianmdisc.gppriors import infer_gp_induced_prior
from bayesianmdisc.gps import (
    GP,
    GaussianProcess,
    IndependentMultiOutputGP,
    condition_gp,
    create_scaled_rbf_gaussian_process,
    optimize_gp_hyperparameters,
)
from bayesianmdisc.io import ProjectDirectory
from bayesianmdisc.models import (
    IsotropicModelLibrary,
    ModelProtocol,
    OrthotropicCANN,
    load_model_state,
    save_model_state,
    trim_model,
)
from bayesianmdisc.normalizingflows import (
    FitNormalizingFlowConfig,
    LoadNormalizingFlowConfig,
    NormalizingFlowProtocol,
    determine_statistical_moments,
    fit_normalizing_flow,
    load_normalizing_flow,
)
from bayesianmdisc.postprocessing.plot import (
    plot_histograms,
    plot_stresses_kawabata,
    plot_stresses_linka,
    plot_stresses_treloar,
)
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed
from bayesianmdisc.statistics.utility import (
    MomentsMultivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
    logarithmic_sum_of_exponentials,
)

data_set_treloar = "treloar"
data_set_kawabata = "kawabata"
data_set_linka = "heart_data_linka"

data_set = data_set_treloar
use_gp_prior = False  # True
retrain_normalizing_flow = True

# Settings
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

# Input/output
current_date = date.today().strftime("%Y%m%d")
if data_set == data_set_treloar:
    input_directory = data_set
    data_reader: DataReaderProtocol = TreloarDataReader(
        input_directory, project_directory, device
    )
elif data_set == data_set_kawabata:
    input_directory = data_set
    data_reader = KawabataDataReader(input_directory, project_directory, device)
elif data_set == data_set_linka:
    input_directory = "heart_data_linka"
    data_reader = LinkaHeartDataReader(input_directory, project_directory, device)


if data_set == data_set_treloar:
    model: ModelProtocol = IsotropicModelLibrary(output_dim=1, device=device)
elif data_set == data_set_kawabata:
    model = IsotropicModelLibrary(output_dim=2, device=device)
elif data_set == data_set_linka:
    model = OrthotropicCANN(device)


relative_noise_stddevs = 5e-2
min_noise_stddev = 1e-3
alpha = 0.0
num_calibration_steps = 2
list_num_wasserstein_iterations = [20_000, 10_000]
list_relative_selection_thressholds = [0.5]
num_flows = 16
relative_width_flow_layers = 4
trim_metric = "rmse"
num_samples_posterior = 4096


output_directory = f"{current_date}_{input_directory}_alpha_{alpha}_nogpprior"
output_subdirectory_name_prior = "prior"
output_subdirectory_name_posterior = "posterior"


def validate_data(
    inputs: DeformationInputs,
    test_cases: TestCases,
    outputs: StressOutputs,
    noise_stddevs: Tensor,
) -> None:
    num_inputs = len(inputs)
    num_test_cases = len(test_cases)
    num_outputs = len(outputs)
    num_noise_stddevs = len(noise_stddevs)

    if (
        num_inputs != num_test_cases
        and num_inputs != num_outputs
        and num_inputs != num_noise_stddevs
    ):
        raise DataError(
            f"""The number of inputs, test cases, outputs and noise standard deviations 
                        is expected to be the same but is {num_inputs}, {num_test_cases}, 
                        {num_outputs} and {num_noise_stddevs}"""
        )


def determine_heteroscedastic_noise() -> Tensor:
    noise_stddevs = relative_noise_stddevs * outputs
    return torch.where(
        noise_stddevs < min_noise_stddev,
        min_noise_stddev,
        noise_stddevs,
    )


@dataclass
class SplittedData:
    inputs_prior: DeformationInputs
    inputs_posterior: DeformationInputs
    test_cases_prior: TestCases
    test_cases_posterior: TestCases
    outputs_prior: StressOutputs
    outputs_posterior: StressOutputs
    noise_stddevs_prior: Tensor
    noise_stddevs_posterior: Tensor


def split_data(
    inputs: DeformationInputs,
    test_cases: TestCases,
    outputs: StressOutputs,
    noise_stddevs: Tensor,
) -> SplittedData:

    def split_treloar_data(
        inputs: DeformationInputs,
        test_cases: TestCases,
        outputs: StressOutputs,
        noise_stddevs: Tensor,
    ) -> SplittedData:
        # Number of data points
        num_points = len(inputs)
        mask_ut = torch.where(test_cases == test_case_identifier_uniaxial_tension)[0]
        num_points_ut = torch.numel(mask_ut)
        mask_ebt = torch.where(test_cases == test_case_identifier_equibiaxial_tension)[
            0
        ]
        num_points_ebt = torch.numel(mask_ebt)

        # Relative indices
        rel_indices_prior_ut = [2, 6, 10, 15, 20]
        rel_indices_prior_ebt = [2, 6, 11]
        rel_indices_prior_ps = [2, 5, 10]
        # Absolute indices
        indices_prior_ut = rel_indices_prior_ut
        start_index = num_points_ut
        indices_prior_ebt = [i + start_index for i in rel_indices_prior_ebt]
        start_index = num_points_ut + num_points_ebt
        indices_prior_ps = [i + start_index for i in rel_indices_prior_ps]
        indices_prior = indices_prior_ut + indices_prior_ebt + indices_prior_ps
        indices_posterior = [i for i in range(num_points) if i not in indices_prior]

        # Data splitting
        inputs_prior = inputs[indices_prior, :]
        inputs_posterior = inputs[indices_posterior, :]
        test_cases_prior = test_cases[indices_prior]
        test_cases_posterior = test_cases[indices_posterior]
        outputs_prior = outputs[indices_prior, :]
        outputs_posterior = outputs[indices_posterior, :]
        noise_stddevs_prior = noise_stddevs[indices_prior]
        noise_stddevs_posterior = noise_stddevs[indices_posterior]

        validate_data(
            inputs_prior, test_cases_prior, outputs_prior, noise_stddevs_prior
        )
        validate_data(
            inputs_posterior,
            test_cases_posterior,
            outputs_posterior,
            noise_stddevs_posterior,
        )
        return SplittedData(
            inputs_prior=inputs_prior,
            inputs_posterior=inputs_posterior,
            test_cases_prior=test_cases_prior,
            test_cases_posterior=test_cases_posterior,
            outputs_prior=outputs_prior,
            outputs_posterior=outputs_posterior,
            noise_stddevs_prior=noise_stddevs_prior,
            noise_stddevs_posterior=noise_stddevs_posterior,
        )

    def split_kawabata_data(
        inputs: DeformationInputs,
        test_cases: TestCases,
        outputs: StressOutputs,
        noise_stddevs: Tensor,
    ) -> SplittedData:
        # Number of data points
        num_points = len(inputs)

        # Indices
        indices_prior = [
            2,
            5,
            10,
            14,
            17,
            22,
            25,
            30,
            34,
            38,
            42,
            46,
            50,
            54,
            57,
            61,
            64,
            68,
            70,
        ]
        indices_posterior = [i for i in range(num_points) if i not in indices_prior]

        # Data splitting
        inputs_prior = inputs[indices_prior, :]
        inputs_posterior = inputs[indices_posterior, :]
        test_cases_prior = test_cases[indices_prior]
        test_cases_posterior = test_cases[indices_posterior]
        outputs_prior = outputs[indices_prior, :]
        outputs_posterior = outputs[indices_posterior, :]
        noise_stddevs_prior = noise_stddevs[indices_prior]
        noise_stddevs_posterior = noise_stddevs[indices_posterior]

        validate_data(
            inputs_prior, test_cases_prior, outputs_prior, noise_stddevs_prior
        )
        validate_data(
            inputs_posterior,
            test_cases_posterior,
            outputs_posterior,
            noise_stddevs_posterior,
        )
        return SplittedData(
            inputs_prior=inputs_prior,
            inputs_posterior=inputs_posterior,
            test_cases_prior=test_cases_prior,
            test_cases_posterior=test_cases_posterior,
            outputs_prior=outputs_prior,
            outputs_posterior=outputs_posterior,
            noise_stddevs_prior=noise_stddevs_prior,
            noise_stddevs_posterior=noise_stddevs_posterior,
        )

    validate_data(inputs, test_cases, outputs, noise_stddevs)
    if data_set == data_set_treloar:
        return split_treloar_data(inputs, test_cases, outputs, noise_stddevs)
    elif data_set == data_set_kawabata:
        return split_kawabata_data(inputs, test_cases, outputs, noise_stddevs)
    else:
        raise DataSetError(f"No implementation for the requested data set {data_set}")


class CombinedPrior:
    def __init__(
        self,
        alpha: float,
        gp_prior: PriorProtocol,
        sparsity_prior: PriorProtocol,
        device: Device,
    ) -> None:
        self._device = device
        self._lower_limit_alpha = 0.0
        self._upper_limit_alpha = 1.0
        self._validate_alpha(alpha)
        self._alpha = torch.tensor(alpha, device=self._device)
        self._inverse_alpha = self._determine_inverse_alpha()
        self._validate_priors(gp_prior, sparsity_prior)
        self._gp_prior = gp_prior
        self._sparsity_prior = sparsity_prior
        self.dim = gp_prior.dim

    def log_prob(self, parameters: Tensor) -> Tensor:
        with torch.no_grad():
            return self._log_prob(parameters)

    def log_prob_with_grad(self, parameters: Tensor) -> Tensor:
        return self._log_prob(parameters)

    def _log_prob(self, parameters: Tensor) -> Tensor:
        log_prob_gp = self._gp_prior.log_prob(parameters)
        log_prob_sparsity = self._sparsity_prior.log_prob(parameters)

        weighted_log_prob_gp = torch.log(self._alpha) + log_prob_gp
        weighted_log_prob_sparsity = torch.log(self._inverse_alpha) + log_prob_sparsity
        log_probs = torch.concat(
            (
                torch.unsqueeze(weighted_log_prob_gp, dim=0),
                torch.unsqueeze(weighted_log_prob_sparsity, dim=0),
            ),
            dim=0,
        )
        return logarithmic_sum_of_exponentials(log_probs)

    def _validate_alpha(self, alpha: float) -> None:
        is_greater_or_equal_lower_limit = alpha >= self._lower_limit_alpha
        is_smaller_or_equal_upper_limit = alpha <= self._upper_limit_alpha
        if not (is_greater_or_equal_lower_limit and is_smaller_or_equal_upper_limit):
            raise CombinedPriorError(
                f"""Alpha is expected to be  >= {self._lower_limit_alpha} 
                                     and <= {self._lower_limit_alpha}, but is {alpha}"""
            )

    def _determine_inverse_alpha(self) -> Tensor:
        return torch.tensor(1.0 - self._alpha, device=device)

    def _validate_priors(
        self, gp_prior: PriorProtocol, sparsity_prior: PriorProtocol
    ) -> None:
        dim_gp_prior = gp_prior.dim
        dim_sparsity_prior = sparsity_prior.dim
        if not dim_gp_prior == dim_sparsity_prior:
            raise CombinedPriorError(
                f"""The GP prior and sparsity prior are expected to have the same dimension, 
                but have {dim_gp_prior} and {dim_sparsity_prior}"""
            )


def sample_from_posterior(
    normalizing_flow: NormalizingFlowProtocol,
) -> tuple[MomentsMultivariateNormal, NPArray]:

    def draw_samples() -> list[Tensor]:
        samples, _ = normalizing_flow.sample(num_samples_posterior)
        return list(samples)

    samples_list = draw_samples()
    return determine_statistical_moments(samples_list)


def plot_stresses(
    model: ModelProtocol, is_model_trimmed: bool, output_directory: str
) -> None:
    def join_output_subdirectory() -> str:
        if is_model_trimmed:
            subdirectory_name = "trimmed_model"
        else:
            subdirectory_name = "untrimmed_model"
        return os.path.join(output_directory, subdirectory_name)

    output_subdirectory = join_output_subdirectory()

    def plot_treloar() -> None:
        plot_stresses_treloar(
            model=cast(IsotropicModelLibrary, model),
            parameter_samples=posterior_samples,
            inputs=inputs.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

        def _plot_kawabata() -> None:
            input_directory = data_set_kawabata
            data_reader = KawabataDataReader(input_directory, project_directory, device)
            output_subdirectory_kawabata = os.path.join(output_subdirectory, "kawabata")

            inputs, test_cases, outputs = data_reader.read()
            isotropic_model = cast(IsotropicModelLibrary, model)
            isotropic_model.set_output_dimension(2)

            plot_stresses_kawabata(
                model=isotropic_model,
                parameter_samples=posterior_samples,
                inputs=inputs.detach().cpu().numpy(),
                outputs=outputs.detach().cpu().numpy(),
                test_cases=test_cases.detach().cpu().numpy(),
                output_subdirectory=output_subdirectory_kawabata,
                project_directory=project_directory,
                device=device,
            )

            isotropic_model.set_output_dimension(1)

        _plot_kawabata()

    def plot_kawabata() -> None:
        plot_stresses_kawabata(
            model=cast(IsotropicModelLibrary, model),
            parameter_samples=posterior_samples,
            inputs=inputs.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

    def plot_linka() -> None:
        plot_stresses_linka(
            model=cast(OrthotropicCANN, model),
            parameter_samples=posterior_samples,
            inputs=inputs.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            test_cases=test_cases.detach().cpu().numpy(),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

    if data_set == data_set_treloar:
        plot_treloar()
    elif data_set == data_set_kawabata:
        plot_kawabata()
    elif data_set == data_set_linka:
        plot_linka()


inputs, test_cases, outputs = data_reader.read()
noise_stddevs = determine_heteroscedastic_noise()
validate_data(inputs, test_cases, outputs, noise_stddevs)

splitted_data = split_data(inputs, test_cases, outputs, noise_stddevs)
inputs_prior = splitted_data.inputs_prior
inputs_posterior = splitted_data.inputs_posterior
test_cases_prior = splitted_data.test_cases_prior
test_cases_posterior = splitted_data.test_cases_posterior
outputs_prior = splitted_data.outputs_prior
outputs_posterior = splitted_data.outputs_posterior
noise_stddevs_prior = splitted_data.noise_stddevs_prior
noise_stddevs_posterior = splitted_data.noise_stddevs_posterior

if retrain_normalizing_flow:
    for step in range(num_calibration_steps):
        output_directory_step = os.path.join(
            output_directory, f"calibration_step_{step}"
        )
        num_parameters = model.get_number_of_active_parameters()

        def determine_prior() -> PriorProtocol | CombinedPrior:
            def init_sparsity_prior() -> PriorProtocol:
                return create_independent_multivariate_gamma_distributed_prior(
                    concentrations=torch.tensor(
                        [0.1 for _ in range(num_parameters)], device=device
                    ),
                    rates=torch.tensor(
                        [10.0 for _ in range(num_parameters)], device=device
                    ),
                    device=device,
                )

            def fit_gp_prior() -> PriorProtocol:
                def create_gaussian_process() -> GaussianProcess:
                    is_single_outut_gp = output_dim == 1
                    jitter = 1e-7

                    def create_single_output_gp() -> GP:
                        gaussian_process = create_scaled_rbf_gaussian_process(
                            mean="zero",
                            input_dims=input_dim,
                            min_inputs=min_inputs,
                            max_inputs=max_inputs,
                            jitter=jitter,
                            device=device,
                        )
                        initial_parameters = torch.tensor(
                            [1.0] + [0.1 for _ in range(input_dim)], device=device
                        )
                        gaussian_process.set_parameters(initial_parameters)
                        return gaussian_process

                    def create_multi_output_gp() -> IndependentMultiOutputGP:
                        gaussian_processes = [
                            create_scaled_rbf_gaussian_process(
                                mean="zero",
                                input_dims=input_dim,
                                min_inputs=min_inputs,
                                max_inputs=max_inputs,
                                jitter=jitter,
                                device=device,
                            )
                            for _ in range(output_dim)
                        ]
                        initial_parameters = torch.tensor(
                            [1.0] + [0.1 for _ in range(input_dim)], device=device
                        )

                        for gaussian_process in gaussian_processes:
                            gaussian_process.set_parameters(initial_parameters)

                        return IndependentMultiOutputGP(
                            gps=tuple(gaussian_processes), device=device
                        )

                    if is_single_outut_gp:
                        return create_single_output_gp()
                    else:
                        return create_multi_output_gp()

                def condition_gaussian_process(
                    inputs: Tensor, outputs: Tensor, noise_stddevs: Tensor
                ) -> None:
                    condition_gp(
                        gaussian_process, inputs, outputs, noise_stddevs, device
                    )

                def determine_prior_moments(
                    samples: Tensor,
                ) -> tuple[MomentsMultivariateNormal, NPArray]:
                    samples_np = samples.detach().cpu().numpy()
                    moments = determine_moments_of_multivariate_normal_distribution(
                        samples_np
                    )
                    return moments, samples_np

                output_subdirectory = os.path.join(
                    output_directory_step, output_subdirectory_name_prior
                )
                min_inputs = torch.amin(inputs, dim=0)
                max_inputs = torch.amax(inputs, dim=0)
                input_dim = inputs.size()[1]
                output_dim = outputs.size()[1]

                gaussian_process = create_gaussian_process()

                optimize_gp_hyperparameters(
                    gaussian_process=gaussian_process,
                    inputs=inputs,
                    outputs=outputs,
                    initial_noise_stddevs=noise_stddevs,
                    num_iterations=int(5e4),
                    learning_rate=1e-3,
                    output_subdirectory=output_subdirectory,
                    project_directory=project_directory,
                    device=device,
                )

                condition_gaussian_process(
                    inputs_prior, outputs_prior, noise_stddevs_prior
                )

                gp_prior = infer_gp_induced_prior(
                    gp=gaussian_process,
                    model=model,
                    prior_type="inverse Gamma",
                    is_mean_trainable=True,
                    inputs=inputs,
                    test_cases=test_cases,
                    num_func_samples=32,
                    resample=True,
                    num_iters_wasserstein=list_num_wasserstein_iterations[step],
                    hiden_layer_size_lipschitz_nn=256,
                    num_iters_lipschitz=5,
                    lipschitz_func_pretraining=True,
                    output_subdirectory=output_subdirectory,
                    project_directory=project_directory,
                    device=device,
                )

                prior_samples = gp_prior.sample(num_samples=4096)
                prior_moments, prior_samples_np = determine_prior_moments(prior_samples)

                plot_histograms(
                    parameter_names=model.parameter_names,
                    true_parameters=tuple(None for _ in range(num_parameters)),
                    moments=prior_moments,
                    samples=prior_samples_np,
                    algorithm_name="gp_prior",
                    output_subdirectory=output_subdirectory,
                    project_directory=project_directory,
                )
                return gp_prior

            if use_gp_prior:
                gp_prior = fit_gp_prior()
                sparsity_prior = init_sparsity_prior()
                return CombinedPrior(alpha, gp_prior, sparsity_prior, device)
            else:
                return init_sparsity_prior()

        def create_likelihood() -> Likelihood:
            return Likelihood(
                model=model,
                relative_noise_stddev=relative_noise_stddevs,
                min_noise_stddev=min_noise_stddev,
                inputs=inputs_posterior,
                test_cases=test_cases_posterior,
                outputs=outputs_posterior,
                device=device,
            )

        prior = determine_prior()
        likelihood = create_likelihood()

        fit_normalizing_flow_config = FitNormalizingFlowConfig(
            likelihood=likelihood,
            prior=prior,
            num_flows=num_flows,
            relative_width_flow_layers=relative_width_flow_layers,
            num_samples=64,
            initial_learning_rate=5e-4,
            final_learning_rate=1e-4,
            num_iterations=100_000,
            deactivate_parameters=False,
            output_subdirectory=output_directory_step,
            project_directory=project_directory,
        )

        normalizing_flow = fit_normalizing_flow(fit_normalizing_flow_config, device)
        posterior_moments, posterior_samples = sample_from_posterior(normalizing_flow)

        output_subdirectory_posterior = os.path.join(
            output_directory_step, output_subdirectory_name_posterior
        )
        plot_histograms(
            parameter_names=model.get_active_parameter_names(),
            true_parameters=tuple(None for _ in range(num_parameters)),
            moments=posterior_moments,
            samples=posterior_samples,
            algorithm_name="nf",
            output_subdirectory=output_subdirectory_posterior,
            project_directory=project_directory,
        )
        plot_stresses(
            model, is_model_trimmed=False, output_directory=output_directory_step
        )

        is_last_step = step == num_calibration_steps - 1
        if not is_last_step:
            trim_model(
                model=model,
                metric=trim_metric,
                relative_thresshold=list_relative_selection_thressholds[step],
                parameter_samples=torch.from_numpy(posterior_samples)
                .type(torch.get_default_dtype())
                .to(device),
                inputs=inputs,
                test_cases=test_cases,
                outputs=outputs,
                output_subdirectory=output_directory_step,
                project_directory=project_directory,
            )
            plot_stresses(
                model, is_model_trimmed=True, output_directory=output_directory_step
            )

            model.reduce_to_activated_parameters()

        save_model_state(model, output_directory_step, project_directory)

else:
    for step in range(num_calibration_steps):
        output_directory_step = os.path.join(
            output_directory, f"calibration_step_{step}"
        )
        is_first_step = step == 0
        if is_first_step:
            input_directory_step = output_directory_step
        else:
            input_step = step - 1
            input_directory_step = os.path.join(
                output_directory, f"calibration_step_{input_step}"
            )

        if not is_first_step:
            load_model_state(model, input_directory_step, project_directory, device)
        num_parameters = model.num_parameters

        load_normalizing_flow_config = LoadNormalizingFlowConfig(
            num_parameters=num_parameters,
            num_flows=num_flows,
            relative_width_flow_layers=relative_width_flow_layers,
            output_subdirectory=output_directory_step,
            project_directory=project_directory,
        )
        normalizing_flow = load_normalizing_flow(load_normalizing_flow_config, device)
        posterior_moments, posterior_samples = sample_from_posterior(normalizing_flow)

        output_subdirectory_posterior = os.path.join(
            output_directory_step, output_subdirectory_name_posterior
        )
        plot_histograms(
            parameter_names=model.parameter_names,
            true_parameters=tuple(None for _ in range(num_parameters)),
            moments=posterior_moments,
            samples=posterior_samples,
            algorithm_name="nf",
            output_subdirectory=output_subdirectory_posterior,
            project_directory=project_directory,
        )
        plot_stresses(
            model, is_model_trimmed=False, output_directory=output_directory_step
        )

        is_last_step = step == num_calibration_steps - 1
        if not is_last_step:
            trim_model(
                model=model,
                metric=trim_metric,
                relative_thresshold=list_relative_selection_thressholds[step],
                parameter_samples=torch.from_numpy(posterior_samples)
                .type(torch.get_default_dtype())
                .to(device),
                inputs=inputs,
                test_cases=test_cases,
                outputs=outputs,
                output_subdirectory=output_directory_step,
                project_directory=project_directory,
            )
            plot_stresses(
                model, is_model_trimmed=True, output_directory=output_directory_step
            )
