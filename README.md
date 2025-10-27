# **UQInModelDiscovery**: Uncertainty Quantification in Model Discovery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17442279.svg)](https://doi.org/10.5281/zenodo.17442182)

The research code **UQInModelDiscovery** provides a framework for uncertainty quantification in model discovery by distilling interpretable material constitutive models from Gaussian process posteriors. In particular, this code repository provides the software for the related scientific publication:

1. [*"Uncertainty quantification in model discovery by distilling interpretable material constitutive models from Gaussian process posteriors"](#uncertainty-quantification-in-model-discovery-by-distilling-interpretable-material-constitutive-models-from-gaussian-process-posteriors)

This code is supposed to be executed in a [*Singularity container*](https://sylabs.io). You can find the [installation instructions](#installation) below.



## Related scientific publications


### Uncertainty quantification in model discovery by distilling interpretable material constitutive models from Gaussian process posteriors

**Citing**:

    @article{anton_UQInModelDiscovery_2025,
        title={Uncertainty quantification in model discovery by distilling interpretable material constitutive models from Gaussian process posteriors},
        author={Anton, David and Wessels, Henning and Römer, Ulrich and Henkes, Alexander and Jorge-Humberto Urrea-Quintero},
        year={2025},
        journal={arXiv preprint},
        doi={}
    }

The results in this publication can be reproduced with the following **script**, which can be found at the top level of this repository:
- *main.py* 

The flag `data_set_label` determines which of the following three numerical test cases is considered:
- `data_set_label_treloar`: Experimental isotropic Treloar dataset
- `data_set_label_anisotropic_synthetic`: Synthetic anisotropic dataset of human cardiac tissue
- `data_set_label_anisotropic`: Experimental anisotropic dataset of human cardiac tissue   

Several other flags are defined at the beginning of the script and at various other points within the script that control the framework's hyperparameters.

> [!IMPORTANT]
> Before running the numerical test cases with experimental data, the experimental data needs to be copied in the inputs directory (see file structure below):
- *Experimental isotropic Treloar dataset*: The experimental data can be copied from [here](https://zenodo.org/records/14995273?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImRmZTZiNWI4LTI1ZGEtNGNjMy1hZDA3LTUwNjE1YmM1MWNmZCIsImRhdGEiOnt9LCJyYW5kb20iOiI3OGFjYTU1OWM0MGU0NzEzYzQyZTkzM2ZiNzViZTFhZCJ9.p7yA7WXVUChxySIYTOXEY3j03DzXEXXbuUizsg5TfxNbFrE1bV8mRKPhnyvETqSRo78R7PAoeOf9Ydi3DK3__Q) and needs to be saved in a directory named *treloar* in the input directory.
- *Experimental anisotropic dataset of human cardiac tissue*: The experimental data can be copied from [here](https://github.com/LivingMatterLab/CANN/blob/main/HEART/input/CANNsHEARTdata_shear05.xlsx) and needs to be saved in a directory named *heart_data_anisotropic* in the input directory.



## Installation


1. For strict separation of input/output data and the source code, the project requires the following file structure:

project_directory \
├── app \
├── input \
└── output

> [!NOTE]
> The output directory is normally created automatically, if it does not already exist. If you are not using any experimental data that needs to be saved in the input directory before the simulation, the input directory is also created automatically.

2. Clone the repository into the *app* directory via:

        git clone https://github.com/david-anton/UQInModelDiscovery .

3. Install the software dependencies. This code is supposed to be executed in a [*Singularity container*](#singularity). In addition, due to the high computational costs, we recommend running the simulations on a GPU. 

4. Run the code.


### Singularity

You can find the singularity definition file in the *.devcontainer* directory. To build the image, navigate to your *project_directory* (see file structure above) and run:

    singularity build uqinmodeldiscovery.sif app/.devcontainer/container.def

Once the image is built, you can run the scripts via:

    singularity run --nv uqinmodeldiscovery.sif python3 <full-path-to-script>/main.py

Please replace `<full-path-to-script>` in the above command according to your file structure.

> [!IMPORTANT]
> You may have to use the *fakreroot* option of singularity if you do not have root rights on your system. In this case, you can try building the image by running the command `singularity build --fakeroot uqinmodeldiscovery.sif app/.devcontainer/container.def`. However, the fakeroot option must be enabled by your system administrator. For further information, please refer to the [Singularity documentation](https://sylabs.io/docs/).



## Citing


If you use this research code, please cite the [related scientific publications](#related-scientic-publications) and the code as follows:

    @misc{anton_codeUQInModelDiscovery_2025,
        title={UQInModelDiscovery: Uncertainty Quantification in Model Discovery},
        author={Anton, David},
        year={2025},
        publisher={Zenodo},
        doi={https://doi.org/10.5281/zenodo.17442182},
        note={Code available from https://github.com/david-anton/UQInModelDiscovery}
    }