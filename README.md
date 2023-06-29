# Recent contributions
This project is based on a fork of the model by Schmittwilken, L. & Maertens, M. (2022).
It has been extended to include a more integrative model of fixational eye movements based on Engbert et al. (2011).
The code has also been revised to support GPU usage through Cupy.
It led to the creation of a poster for an oral presentation in 2023.  
See [Active Vision Model for Edge Detection.pdf](./Active%20Vision%20Model%20for%20Edge%20Detection.pdf) by Jérémy Faggion and Vanessa Martin, under the supervision of Laurent Pezard.

### Installation Guide
This project uses several Python packages which need to be installed before you can run the code. We use Anaconda to manage these packages, which we have exported into an environment.yml file.

If you haven't already, [install Anaconda](https://docs.anaconda.com/anaconda/install/) on your system.
Clone the repository to your local machine.
Navigate to the project directory and run the following command to create a new environment from the environment.yml file:
```
conda env create -f environment.yml
```
This will create a new Conda environment with all the necessary packages.

Please note that this project uses the cupy library, which requires the CUDA toolkit for GPU-accelerated computing. Make sure you have the CUDA toolkit version 11.7 installed on your machine. If you don't have it installed, you can download it from the [official NVIDIA website](https://developer.nvidia.com/cuda-toolkit-archive). Follow the instructions there to install the CUDA toolkit.

Once all the packages are installed and CUDA is set up, you should be able to run the project without any issues. If you run into any problems, feel free to open an issue on this repository.

# ORIGINAL README
This is the code used to produce the results and visualizations published in

Schmittwilken, L. & Maertens, M. (2022). Fixational eye movements enable robust edge detection. Journal of Vision, 22(5). [doi:10.1167/jov.22.8.5](https://doi.org/10.1167/jov.22.8.5)

## Description
The repository contains the following:

- The data from the psychophysical experiment of Betz et al. (2015) and the Contour Image Database by Grigorescu et al. (2003) that is used as test cases of the model: [databases](databases)

- Two Jupyter-Notebooks with a step-by-step guide through the proposed active edge detection model [active_edge-model.ipynb](jupyter-notebooks/active_edge-model.ipynb) and a demonstration of how spatial edge models work [spatial_edge-models.ipynb](jupyter-notebooks/spatial_edge-models.ipynb)

- Code to create the results shown in the paper: [simulations](simulations). To reproduce the results of test case 1 (edge detection in narrowband noise), run [main_case1.py](simulations/main_case1.py). To reproduce the results of test case 2 (contour detection in natural images), run [main_case2.py](simulations/main_case2.py)

- Code to create the visualizations from the manuscript: [visualize_results](visualize_results). In order to re-create the visualizations, first run the simulations to produce the respective results.

## Authors and acknowledgment
Code written by Lynn Schmittwilken (l.schmittwilken@tu-berlin.de)

## References
Betz, T., Shapley, R., Wichmann, F. A., & Maertens, M. (2015a). Noise masking of White's illusion exposes the weakness of current spatial filtering models of lightness perception. Journal of Vision, 15(14), 1, doi:10.1167/15.14.1

Grigorescu, C., Petkov, N., & Westenberg, M. A. (2003). Contour detection based on nonclassical receptive field inhibition. IEEE Transactions on Image Processing, 12(7), 729–739, doi:10.1109/TIP.2003.814250

