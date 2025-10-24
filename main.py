import os
from datetime import date
from time import perf_counter
from typing import cast

import torch

from bayesianmdisc.bayes.distributions import DistributionProtocol
from bayesianmdisc.customtypes import GPModel, NPArray, Tensor
from bayesianmdisc.data import (
    AnisotropicHeartDataSet,
    AnisotropicHeartDataSetGenerator,
    DataSetProtocol,
    TreloarDataSet,
    add_noise_to_data,
    determine_heteroscedastic_noise,
    validate_data,
)
from bayesianmdisc.datasettings import (
    assemble_input_mask_for_treloar_data,
    assemble_input_masks_for_anisotropic_data,
    create_four_terms_anisotropic_model_parameters,
    data_set_label_anisotropic,
    data_set_label_anisotropic_synthetic,
    data_set_label_treloar,
)
from bayesianmdisc.errors import MainError
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
    IsotropicModel,
    ModelProtocol,
    OrthotropicCANN,
    OutputSelectorAnisotropic,
    OutputSelectorProtocol,
    OutputSelectorTreloar,
    ParameterNames,
    create_isotropic_model,
    load_model_state,
    save_model_state,
    select_model_through_sobol_sensitivity_analysis,
)
from bayesianmdisc.normalizingflows import NormalizingFlowDistribution
from bayesianmdisc.parameterdistillation import (
    distill_parameter_distribution_from_gp,
    load_normalizing_flow_parameter_distribution,
    save_normalizing_flow_parameter_distribution,
)
from bayesianmdisc.postprocessing.plot import (
    TrueParameters,
    plot_gp_stresses_anisotropic,
    plot_gp_stresses_treloar,
    plot_histograms,
    plot_model_stresses_anisotropic,
    plot_model_stresses_treloar,
    plot_sobol_indice_paths_anisotropic,
    plot_sobol_indice_paths_treloar,
)
from bayesianmdisc.settings import Settings, get_device, set_default_dtype, set_seed
from bayesianmdisc.utility import from_torch_to_numpy

data_set_label = data_set_label_treloar
retrain_models = True

# Settings
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

current_date = date.today().strftime("%Y%m%d")
if data_set_label == data_set_label_treloar:
    input_directory = data_set_label
    data_set: DataSetProtocol = TreloarDataSet(
        input_directory, project_directory, device
    )

    strain_energy_function_type = "library"
    model: ModelProtocol = create_isotropic_model(
        strain_energy_function_type=strain_energy_function_type,
        output_dim=1,
        device=device,
    )

    relative_noise_stddevs = 5e-2
    min_absolute_noise_stddev = 1e-2
    list_num_wasserstein_iterations = [20_000, 10_000]
    if strain_energy_function_type == "library":
        total_sobol_index_thresshold = 1e-4
    elif strain_energy_function_type == "cann":
        total_sobol_index_thresshold = 1e-2
elif data_set_label == data_set_label_anisotropic:
    input_directory = data_set_label
    data_set = AnisotropicHeartDataSet(
        input_directory=input_directory,
        file_name="CANNsHEARTdata_shear05.xlsx",
        project_directory=project_directory,
        device=device,
    )
    num_points_per_test_case = 11

    use_only_squared_anisotropic_invariants = True
    model = OrthotropicCANN(device, use_only_squared_anisotropic_invariants)

    relative_noise_stddevs = 5e-2
    min_absolute_noise_stddev = 1e-2
    list_num_wasserstein_iterations = [20_000, 10_000]
    total_sobol_index_thresshold = 1e-2
elif data_set_label == data_set_label_anisotropic_synthetic:
    input_directory = data_set_label
    file_name = "CANNsHEARTdata_synthetic.xlsx"
    num_points_per_test_case = 11
    use_only_squared_anisotropic_invariants = True

    model_data_generation = OrthotropicCANN(
        device, use_only_squared_anisotropic_invariants
    )
    four_terms_model_parameters = create_four_terms_anisotropic_model_parameters()
    model_data_generation.reduce_model_to_parameter_names(
        four_terms_model_parameters.names
    )
    data_generator = AnisotropicHeartDataSetGenerator(
        model=model_data_generation,
        parameters=four_terms_model_parameters.values,
        num_point_per_test_case=num_points_per_test_case,
        file_name=file_name,
        output_directory=input_directory,
        project_directory=project_directory,
        device=device,
    )
    data_generator.generate()
    data_set = AnisotropicHeartDataSet(
        input_directory=input_directory,
        file_name=file_name,
        project_directory=project_directory,
        device=device,
    )

    model = OrthotropicCANN(device, use_only_squared_anisotropic_invariants)

    relative_noise_stddevs = 5e-2
    min_absolute_noise_stddev = 1e-2
    list_num_wasserstein_iterations = [20_000, 10_000]
    total_sobol_index_thresshold = 1e-2

num_samples_parameter_distribution = 8192
num_samples_factor_sensitivity_analysis = 4096


output_directory = f"{current_date}_{data_set_label}"
output_subdirectory_name_gp = "gp"
output_subdirectory_name_parameters = "parameters"
output_subdirectory_name_sensitivities = "sensitivity_analysis"


def plot_parameter_histograms(
    parameter_names: tuple[str, ...],
    true_parameters: TrueParameters,
    parameter_samples: NPArray,
    subdirectory_name: str,
    output_directory: str,
) -> None:

    def join_output_subdirectory() -> str:
        return os.path.join(output_directory, subdirectory_name)

    output_subdirectory = join_output_subdirectory()
    plot_histograms(
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        samples=parameter_samples,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
    )


def plot_gp_stresses(
    gaussian_process: GaussianProcess,
    data_set_label: str,
    output_subdirectory: str,
) -> None:

    def plot_treloar() -> None:
        plot_gp_stresses_treloar(
            gaussian_process=gaussian_process,
            inputs=from_torch_to_numpy(inputs),
            outputs=from_torch_to_numpy(outputs),
            test_cases=from_torch_to_numpy(test_cases),
            noise_stddevs=from_torch_to_numpy(noise_stddevs),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

    def plot_anisotropic() -> None:
        plot_gp_stresses_anisotropic(
            gaussian_process=gaussian_process,
            inputs=from_torch_to_numpy(inputs),
            outputs=from_torch_to_numpy(outputs),
            test_cases=from_torch_to_numpy(test_cases),
            noise_stddevs=from_torch_to_numpy(noise_stddevs),
            num_points_per_test_case=num_points_per_test_case,
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

    if data_set_label == data_set_label_treloar:
        plot_treloar()
    elif (
        data_set_label == data_set_label_anisotropic
        or data_set_label == data_set_label_anisotropic_synthetic
    ):
        plot_anisotropic()


def plot_model_stresses(
    model: ModelProtocol,
    parameter_samples: NPArray,
    data_set_label: str,
    subdirectory_name: str,
    output_directory: str,
) -> None:

    def join_output_subdirectory() -> str:
        return os.path.join(output_directory, subdirectory_name)

    output_subdirectory = join_output_subdirectory()

    def plot_treloar() -> None:
        plot_model_stresses_treloar(
            model=cast(IsotropicModel, model),
            parameter_samples=parameter_samples,
            inputs=from_torch_to_numpy(inputs),
            outputs=from_torch_to_numpy(outputs),
            test_cases=from_torch_to_numpy(test_cases),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )

    def plot_anisotropic() -> None:
        plot_four_term_model = True
        plot_model_stresses_anisotropic(
            model=cast(OrthotropicCANN, model),
            parameter_samples=parameter_samples,
            inputs=from_torch_to_numpy(inputs),
            test_cases=from_torch_to_numpy(test_cases),
            outputs=from_torch_to_numpy(outputs),
            num_points_per_test_case=num_points_per_test_case,
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
            device=device,
            plot_four_term_model=plot_four_term_model,
        )

    if data_set_label == data_set_label_treloar:
        plot_treloar()
    elif (
        data_set_label == data_set_label_anisotropic
        or data_set_label == data_set_label_anisotropic_synthetic
    ):
        plot_anisotropic()


def plot_sobol_indices_results(
    relevant_parameter_indices: list[int],
    data_set_label: str,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    if data_set_label == data_set_label_treloar:
        plot_sobol_indice_paths_treloar(
            relevant_parameter_indices=relevant_parameter_indices,
            inputs=from_torch_to_numpy(inputs),
            test_cases=from_torch_to_numpy(test_cases),
            outputs=from_torch_to_numpy(outputs),
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
        )
    elif (
        data_set_label == data_set_label_anisotropic
        or data_set_label == data_set_label_anisotropic_synthetic
    ):
        plot_sobol_indice_paths_anisotropic(
            relevant_parameter_indices=relevant_parameter_indices,
            num_points_per_testcase=num_points_per_test_case,
            output_subdirectory=output_subdirectory,
            project_directory=project_directory,
        )


def select_reduced_parameter_samples(
    parameter_samples_full: NPArray,
    parameter_names_full: ParameterNames,
    parameter_names_reduced,
) -> NPArray:
    reduced_indices = [
        parameter_names_full.index(parameter_name)
        for parameter_name in parameter_names_reduced
    ]
    return parameter_samples_full[:, reduced_indices]


inputs, test_cases, outputs = data_set.read_data()
noise_stddevs = determine_heteroscedastic_noise(
    relative_noise_stddevs, min_absolute_noise_stddev, outputs
)
if data_set_label == data_set_label_anisotropic_synthetic:
    outputs = add_noise_to_data(noise_stddevs, outputs, device)


validate_data(inputs, test_cases, outputs, noise_stddevs)
num_discovery_steps = len(list_num_wasserstein_iterations)

if retrain_models:
    start_time = perf_counter()
    for step in range(num_discovery_steps):
        is_first_step = step == 0
        is_last_step = step == num_discovery_steps - 1
        output_directory_step = os.path.join(output_directory, f"discovery_step_{step}")
        output_subdirectory_gp = os.path.join(
            output_directory_step, output_subdirectory_name_gp
        )
        output_subdirectory_parameters = os.path.join(
            output_directory_step, output_subdirectory_name_parameters
        )
        output_subdirectory_sensitivities = os.path.join(
            output_directory_step, output_subdirectory_name_sensitivities
        )

        def create_gp() -> GaussianProcess:

            def create_single_output_gp(input_mask: Tensor | None) -> GP:
                min_inputs = torch.amin(inputs, dim=0)
                max_inputs = torch.amax(inputs, dim=0)
                jitter = 1e-7
                gp_mean = "zero"
                if input_mask is not None:
                    input_dim = int(torch.sum(input_mask).item())
                else:
                    input_dim = inputs.size()[1]

                gaussian_process = create_scaled_rbf_gaussian_process(
                    mean=gp_mean,
                    input_dim=input_dim,
                    min_inputs=min_inputs,
                    max_inputs=max_inputs,
                    input_mask=input_mask,
                    jitter=jitter,
                    device=device,
                )
                initial_parameters_output_scale = [1.0]
                initial_parameters_length_scale = [0.1 for _ in range(input_dim)]
                initial_parameters = torch.tensor(
                    initial_parameters_output_scale + initial_parameters_length_scale,
                    device=device,
                )
                gaussian_process.set_parameters(initial_parameters)
                return gaussian_process

            if data_set_label == data_set_label_treloar:
                input_mask = assemble_input_mask_for_treloar_data(device)
                return create_single_output_gp(input_mask)
            elif (
                data_set_label == data_set_label_anisotropic
                or data_set_label == data_set_label_anisotropic_synthetic
            ):
                input_masks = assemble_input_masks_for_anisotropic_data(device)
                gaussian_processes = [
                    create_single_output_gp(input_mask) for input_mask in input_masks
                ]
                return IndependentMultiOutputGP(
                    gps=tuple(gaussian_processes), device=device
                )
            else:
                raise MainError(f"Unknwon dataset label: {data_set_label}")

        def select_gp_prior() -> None:
            num_iterations = int(1e4)
            learning_rate = 2e-1
            if data_set_label == data_set_label_treloar:
                factor_length_scales = 0.8
            elif data_set_label == data_set_label_anisotropic:
                factor_length_scales = 0.6
            elif data_set_label == data_set_label_anisotropic_synthetic:
                factor_length_scales = 0.6

            def optimize_hyperparameters() -> None:
                return optimize_gp_hyperparameters(
                    gaussian_process=gaussian_process,
                    inputs=inputs,
                    outputs=outputs,
                    initial_noise_stddevs=noise_stddevs,
                    num_iterations=num_iterations,
                    learning_rate=learning_rate,
                    output_subdirectory=output_subdirectory_gp,
                    project_directory=project_directory,
                    device=device,
                )

            def scale_length_scales(factor: float) -> None:
                def _scale_length_scales_of_one_gp(gp: GaussianProcess):
                    parameters_dict = gp.get_named_parameters()
                    length_scale = parameters_dict["length_scale"]
                    parameters_dict["length_scale"] = factor * length_scale
                    scaled_parameters = list(parameters_dict.values())
                    scaled_parameters = [
                        parameters.reshape((-1,)) for parameters in scaled_parameters
                    ]
                    gp.set_parameters(torch.concat(scaled_parameters).to(device))

                if gaussian_process.num_gps == 1:
                    _scale_length_scales_of_one_gp(gaussian_process)
                else:
                    for gp in gaussian_process.gps.models:
                        _gp = cast(GPModel, gp)
                        _scale_length_scales_of_one_gp(_gp)

            optimize_hyperparameters()
            scale_length_scales(factor_length_scales)
            print(f"Scaled GP parameters: {gaussian_process.get_named_parameters()}")

        def infer_gp_posterior() -> None:
            condition_gp(
                gaussian_process=gaussian_process,
                inputs=inputs,
                outputs=outputs,
                noise_stddevs=noise_stddevs,
                device=device,
            )
            plot_gp_stresses(
                gaussian_process=gaussian_process,
                data_set_label=data_set_label,
                output_subdirectory=output_subdirectory_gp,
            )

        def distill_parameter_distribution() -> DistributionProtocol:
            num_func_samples = 32
            num_points_per_test_case = 32
            num_iters_lipschitz = 10
            num_layers_lipschitz_nn = 2
            relative_width_lipschitz_nn = 4

            if data_set_label == data_set_label_treloar:
                lipschitz_penalty_coefficient = 10.0
                data_set_treloar = cast(TreloarDataSet, data_set)
                inputs_extraction, test_cases_extraction = (
                    data_set_treloar.generate_uniform_inputs(num_points_per_test_case)
                )
                output_selector: OutputSelectorProtocol = OutputSelectorTreloar(
                    test_cases_extraction, cast(IsotropicModel, model), device
                )
            elif data_set_label == data_set_label_anisotropic:
                lipschitz_penalty_coefficient = 100.0
                data_set_anisotropic = cast(AnisotropicHeartDataSet, data_set)
                inputs_extraction, test_cases_extraction = (
                    data_set_anisotropic.generate_uniform_inputs(
                        num_points_per_test_case
                    )
                )
                output_selector = OutputSelectorAnisotropic(
                    test_cases_extraction, cast(OrthotropicCANN, model), device
                )
            elif data_set_label == data_set_label_anisotropic_synthetic:
                lipschitz_penalty_coefficient = 100.0
                data_set_anisotropic = cast(AnisotropicHeartDataSet, data_set)
                inputs_extraction, test_cases_extraction = (
                    data_set_anisotropic.generate_uniform_inputs(
                        num_points_per_test_case
                    )
                )
                output_selector = OutputSelectorAnisotropic(
                    test_cases_extraction, cast(OrthotropicCANN, model), device
                )

            distribution = distill_parameter_distribution_from_gp(
                gp=gaussian_process,
                model=model,
                output_selector=output_selector,
                distribution_type="normalizing flow",
                inputs=inputs_extraction,
                test_cases=test_cases_extraction,
                num_func_samples=num_func_samples,
                lipschitz_penalty_coefficient=lipschitz_penalty_coefficient,
                num_iters_wasserstein=list_num_wasserstein_iterations[step],
                num_layers_lipschitz_nn=num_layers_lipschitz_nn,
                relative_width_lipschitz_nn=relative_width_lipschitz_nn,
                num_iters_lipschitz=num_iters_lipschitz,
                output_subdirectory=output_subdirectory_parameters,
                project_directory=project_directory,
                device=device,
            )
            if isinstance(distribution, NormalizingFlowDistribution):
                save_normalizing_flow_parameter_distribution(
                    distribution, output_directory_step, project_directory, device
                )
            return distribution

        gaussian_process = create_gp()
        select_gp_prior()
        infer_gp_posterior()
        parameter_distribution = distill_parameter_distribution()

        num_parameters_full = model.num_parameters
        parameter_names_full = model.parameter_names
        true_model_parameters_full = tuple(None for _ in range(num_parameters_full))
        parameter_samples_full = (
            parameter_distribution.sample(num_samples_parameter_distribution)
            .detach()
            .cpu()
            .numpy()
        )

        plot_parameter_histograms(
            parameter_names=parameter_names_full,
            true_parameters=true_model_parameters_full,
            parameter_samples=parameter_samples_full,
            subdirectory_name="full",
            output_directory=output_subdirectory_parameters,
        )
        plot_model_stresses(
            model=model,
            parameter_samples=parameter_samples_full,
            data_set_label=data_set_label,
            subdirectory_name="model_full",
            output_directory=output_directory_step,
        )

        select_model_through_sobol_sensitivity_analysis(
            model=model,
            parameter_distribution=parameter_distribution,
            total_sobol_index_thresshold=total_sobol_index_thresshold,
            num_samples_factor=num_samples_factor_sensitivity_analysis,
            data_set_label=data_set_label,
            inputs=inputs,
            test_cases=test_cases,
            output_subdirectory=output_subdirectory_sensitivities,
            project_directory=project_directory,
            device=device,
        )

        num_parameters_reduced = model.get_number_of_active_parameters()
        parameter_names_reduced = model.get_active_parameter_names()
        true_model_parameters_reduced = tuple(
            None for _ in range(num_parameters_reduced)
        )
        parameter_samples_reduced = select_reduced_parameter_samples(
            parameter_samples_full=parameter_samples_full,
            parameter_names_full=parameter_names_full,
            parameter_names_reduced=parameter_names_reduced,
        )

        plot_sobol_indices_results(
            relevant_parameter_indices=model.get_active_parameter_indices(),
            data_set_label=data_set_label,
            output_subdirectory=output_subdirectory_sensitivities,
            project_directory=project_directory,
        )
        plot_parameter_histograms(
            parameter_names=parameter_names_reduced,
            true_parameters=true_model_parameters_reduced,
            parameter_samples=parameter_samples_reduced,
            subdirectory_name="reduced",
            output_directory=output_subdirectory_parameters,
        )
        plot_model_stresses(
            model=model,
            parameter_samples=parameter_samples_full,
            data_set_label=data_set_label,
            subdirectory_name="model_selected",
            output_directory=output_directory_step,
        )

        if is_last_step:
            model.reset_parameter_deactivations()
        else:
            model.reduce_to_activated_parameters()

        save_model_state(model, output_directory_step, project_directory)

    end_time = perf_counter()
    run_time = end_time - start_time
    print(f"Total run time: {run_time} s")
else:
    for step in range(num_discovery_steps):
        is_first_step = step == 0
        is_last_step = step == num_discovery_steps - 1
        output_directory_step = os.path.join(output_directory, f"discovery_step_{step}")
        output_subdirectory_parameters = os.path.join(
            output_directory_step, output_subdirectory_name_parameters
        )
        output_subdirectory_sensitivities = os.path.join(
            output_directory_step, output_subdirectory_name_sensitivities
        )

        if not is_first_step:
            input_step = step - 1
            input_directory_step = os.path.join(
                output_directory, f"discovery_step_{input_step}"
            )
            load_model_state(model, input_directory_step, project_directory, device)

        def load_parameter_distribution() -> DistributionProtocol:
            return load_normalizing_flow_parameter_distribution(
                model=model,
                output_subdirectory=output_directory_step,
                project_directory=project_directory,
                device=device,
            )

        parameter_distribution = load_parameter_distribution()

        num_parameters_full = model.num_parameters
        parameter_names_full = model.parameter_names
        true_model_parameters_full = tuple(None for _ in range(num_parameters_full))
        parameter_samples_full = (
            parameter_distribution.sample(num_samples_parameter_distribution)
            .detach()
            .cpu()
            .numpy()
        )

        plot_parameter_histograms(
            parameter_names=parameter_names_full,
            true_parameters=true_model_parameters_full,
            parameter_samples=parameter_samples_full,
            subdirectory_name="full",
            output_directory=output_subdirectory_parameters,
        )
        plot_model_stresses(
            model=model,
            parameter_samples=parameter_samples_full,
            data_set_label=data_set_label,
            subdirectory_name="model_full",
            output_directory=output_directory_step,
        )

        select_model_through_sobol_sensitivity_analysis(
            model=model,
            parameter_distribution=parameter_distribution,
            total_sobol_index_thresshold=total_sobol_index_thresshold,
            num_samples_factor=num_samples_factor_sensitivity_analysis,
            data_set_label=data_set_label,
            inputs=inputs,
            test_cases=test_cases,
            output_subdirectory=output_subdirectory_sensitivities,
            project_directory=project_directory,
            device=device,
        )

        num_parameters_reduced = model.get_number_of_active_parameters()
        parameter_names_reduced = model.get_active_parameter_names()
        true_model_parameters_reduced = tuple(
            None for _ in range(num_parameters_reduced)
        )
        parameter_samples_reduced = select_reduced_parameter_samples(
            parameter_samples_full=parameter_samples_full,
            parameter_names_full=parameter_names_full,
            parameter_names_reduced=parameter_names_reduced,
        )

        plot_sobol_indices_results(
            relevant_parameter_indices=model.get_active_parameter_indices(),
            data_set_label=data_set_label,
            output_subdirectory=output_subdirectory_sensitivities,
            project_directory=project_directory,
        )

        plot_parameter_histograms(
            parameter_names=parameter_names_reduced,
            true_parameters=true_model_parameters_reduced,
            parameter_samples=parameter_samples_reduced,
            subdirectory_name="reduced",
            output_directory=output_subdirectory_parameters,
        )
        plot_model_stresses(
            model=model,
            parameter_samples=parameter_samples_full,
            data_set_label=data_set_label,
            subdirectory_name="model_selected",
            output_directory=output_directory_step,
        )

        if is_last_step:
            model.reset_parameter_deactivations()
