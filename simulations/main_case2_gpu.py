"""
This script runs test case 2 of our paper.
We test the performance of the (control) models on the contour database by
Grigorescu et al. (2003). The database contains 40 grey-level images of size
512x512 pixels, in PGM format. Each image represents a natural image scene and is
accompanied by an associated ground truth contour map drawn by a human.
We quantify the model performance by correlating our model outputs with the
ground truth contour maps that were drawn by humans.

Last update on 04.06.2022
@author: lynnschmittwilken

Added Nvidia GPU support
Last update on 29.06.2023
@author: jeremy-fgn
"""

import cv2
import numpy as np
import time
import os
import glob
import pickle
import cupy as cp
import matplotlib.pyplot as plt

import parameters as params
from functions import (
    octave_intervals,
    create_dog,
    create_tfilt_gpu,
    create_brownian_drift,
    create_integrative_drift_microsaccades,
    run_active_model_gpu,
    quantify_edges,
    thicken_edges,
    remove_borders,
    check_gpu_support,
)


mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
cache = cp.fft.config.get_plan_cache()
cache.set_size(2)


def load_and_preprocess_images(img_names, input_path, edge_thickness, rb):
    stimuli = []
    gt_templates = []

    for img_id in range(len(img_names)):
        img_name = img_names[img_id]

        # Import and preprocess stimulus
        stimulus_int = cv2.imread(input_path + img_name + ".pgm").astype(np.uint16)
        stimulus_int = cv2.cvtColor(stimulus_int, cv2.COLOR_BGR2GRAY).astype(np.uint16)
        stimulus_float = np.float16(stimulus_int / 255.0)

        # Import and preprocess ground truth
        gt_template_int = cv2.imread(
            input_path + "gt/" + img_name + "_gt_binary.pgm"
        ).astype(np.uint16)
        gt_template_int = cv2.cvtColor(gt_template_int, cv2.COLOR_BGR2GRAY).astype(
            np.uint16
        )
        gt_template_float = np.array(gt_template_int / 255.0)
        gt_template_float = np.abs(gt_template_float - 1.0)

        gt_template_float = thicken_edges(gt_template_float, edge_thickness).astype(
            np.float16
        )
        gt_template_float = remove_borders(gt_template_float, rb).astype(np.float16)

        stimuli.append(stimulus_float)
        gt_templates.append(gt_template_float)

    return stimuli, gt_templates


def create_dog_filters(n_filters, stimulus_size, ppd, smax):
    fs = cp.fft.fftshift(cp.fft.fftfreq(stimulus_size, d=1.0 / ppd)).astype(cp.float32)

    sigmas = cp.float32(octave_intervals(n_filters) * smax)
    fx, fy = cp.meshgrid(fs, fs)
    dogs = [
        cp.expand_dims(
            create_dog(fx, fy, sigmas[i], 2.0 * sigmas[i]).astype(cp.float32), -1
        )
        for i in range(n_filters)
    ]

    dogs = cp.array(
        dogs,
        dtype=cp.float32,
    ).reshape(n_filters, stimulus_size, stimulus_size, 1)
    fs = None
    sigmas = None
    fx, fy = None, None
    return dogs


def generate_noisy_stimulus(stimulus, current_s_noise):
    noise = np.random.normal(
        0.0, current_s_noise, [stimulus_size, stimulus_size]
    ).astype(np.float16)
    stimulus += noise
    stimulus[stimulus <= 0.0] = 0.0
    stimulus[stimulus >= 1.0] = 1.0
    return stimulus


def create_temporal_filter(T, pps):
    # Define temporal frequency axis:
    nT = int(T * pps + 1)
    tff = cp.fft.fftshift(cp.fft.fftfreq(nT, d=1.0 / pps)).astype(cp.float32)
    # Temporal filter:
    tfilt = create_tfilt_gpu(tff)
    # For performing 3d fft in space (x, y) and time (t):
    tfilt = cp.abs(cp.expand_dims(tfilt, (0, 1)))

    tff = None
    return tfilt


def generate_paths_wrapper(method_choice, T, pps, ppd, D):
    if method_choice == "integrative_no_MS":
        return create_integrative_drift_microsaccades(
            T, pps, ppd, D, microsaccade_threshold=100
        )
    elif method_choice == "integrative_with_MS":
        return create_integrative_drift_microsaccades(
            T, pps, ppd, D, microsaccade_threshold=7.3
        )
    elif method_choice == "integrative_weak_attractor":
        return create_integrative_drift_microsaccades(
            T, pps, ppd, D, microsaccade_threshold=100, slope=0.3
        )
    else:
        return create_brownian_drift(T, pps, ppd, D)


# Constants
VALID_IMAGE_INDEX_RANGE = range(40)
EYE_MOVEMENT_METHODS = {
    1: "simple",
    2: "integrative_no_MS",
    3: "integrative_with_MS",
    4: "integrative_weak_attractor",
}

EDGE_DETECTION_MODELS = {1: "ST_M_N", 2: "S_V_N"}


def prompt_for_index(prompt, valid_range):
    """Prompts the user for an integer and checks it against a valid range."""
    value = int(input(prompt))
    if value not in valid_range:
        print(
            f"Invalid input. Please enter a number between {valid_range.start} and {valid_range.stop - 1}."
        )
        exit()
    return value


def prompt_for_float(prompt):
    """Prompts the user for a float."""
    return float(input(prompt))


def prompt_for_choice(prompt, choices_dict):
    """Prompts the user for a choice and maps it to a value in a dictionary."""
    choice = int(input(prompt))
    return choices_dict.get(choice, choices_dict[1])  # Default to the first choice


if __name__ == "__main__":
    if not check_gpu_support():
        raise RuntimeError("GPU support is not available.")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(project_root, "data")
    output_dir = os.path.join(results_path, "output_images")
    os.makedirs(output_dir, exist_ok=True)

    # Input prompts
    print("Please enter the following values:")
    start = prompt_for_index(
        "Enter the start image index (starts at 0): ", VALID_IMAGE_INDEX_RANGE
    )
    end = prompt_for_index("Enter the end image index: ", VALID_IMAGE_INDEX_RANGE)
    eye_movement_method = prompt_for_choice(
        "Enter the eye movement method (1=simple, 2=integrative no MS, 3=integrative with MS, 4=integrative weak attractor): ",
        EYE_MOVEMENT_METHODS,
    )
    edge_detection_model = prompt_for_choice(
        "Finally, enter the edge detection model (1=ST_M_N, 2=S_V_N): ",
        EDGE_DETECTION_MODELS,
    )

    # Simulation Parameters
    parameters = {
        "eye_movement_method": eye_movement_method,
        "images_to_be_processed": np.arange(start, end + 1, 1, dtype=np.uint16),
        "input_path": os.path.join(
            project_root,
            "databases/grigorescu2003_contour_image_database/images/",
        ),
        "s_noise_values": np.array([0, 0.1], dtype=np.float16),
        "ppd": np.float16(params.ppd),
        "n_trials": np.uint16(params.n_trials),
        "D": round((params.D), 6),
        "T": np.float16(params.T),
        "pps": np.uint16(params.pps),
        "n_filters": np.uint16(params.n_filters),
        "smax": np.float16(params.smax),
        "edge_thickness": np.uint16(params.edge_thickness * params.ppd),
        "remove_b": int(params.remove_b * params.ppd),
    }

    # Create img_names list
    img_names = glob.glob(parameters["input_path"] + "*.pgm")
    img_names = [
        f[0:-4]
        for f in os.listdir(parameters["input_path"])
        if os.path.isfile(os.path.join(parameters["input_path"], f))
    ]
    img_names = [img_names[i] for i in parameters["images_to_be_processed"]]
    # Load one exemplary input image to get stimulus size
    stimulus = cv2.imread(parameters["input_path"] + img_names[0] + ".pgm")
    stimulus_size = np.uint16(stimulus.shape[0])
    # Visual extent for the input stimuli:
    im_size_h = stimulus.shape[0] / parameters["ppd"]
    im_size_w = stimulus.shape[1] / parameters["ppd"]
    visual_extent = [
        -im_size_h / 2,
        im_size_h / 2,
        -im_size_w / 2,
        im_size_w / 2,
    ]  # in deg
    rb = parameters["remove_b"]
    vextent_adapt = [
        visual_extent[0] + rb,
        visual_extent[1] - rb,
        visual_extent[2] + rb,
        visual_extent[3] - rb,
    ]
    # Load and preprocess images
    stimuli, gt_templates = load_and_preprocess_images(
        img_names,
        parameters["input_path"],
        parameters["edge_thickness"],
        parameters["remove_b"],
    )

    # Create filters
    dogs = create_dog_filters(
        parameters["n_filters"], stimulus_size, parameters["ppd"], parameters["smax"]
    )
    tfilt = create_temporal_filter(parameters["T"], parameters["pps"])

    # Initialize variables that will live on the GPU
    model_stimulus_gpu = cp.zeros((stimulus_size, stimulus_size), dtype=cp.float16)
    model_output_gpu = cp.zeros((stimulus_size, stimulus_size), dtype=cp.float32)
    drift_int_gpu = cp.zeros((2, parameters["pps"]), dtype=cp.float16)

    iterations = len(parameters["images_to_be_processed"]) * len(
        parameters["s_noise_values"]
    )
    results_list = []
    iteration = 0

    for img_id in range(len(img_names)):
        img_name = img_names[img_id]
        stimulus_float = stimuli[img_id]
        back_lum = np.mean(stimulus_float[0])
        gt_template = gt_templates[img_id]
        print(
            "Processing image "
            + str(img_id + 1)
            + "/"
            + str(len(img_names))
            + " ("
            + img_name
            + ")"
        )
        for s_noise_value in parameters["s_noise_values"]:
            current_s_noise = s_noise_value
            print("Gaussian noise level: " + str(current_s_noise))
            start = time.time()
            iteration += 1
            best_correlation = None
            best_output_image = None
            correlation_scores = []

            for trial in range(parameters["n_trials"]):
                if current_s_noise > 0:
                    stimulus_float = generate_noisy_stimulus(
                        stimulus_float, current_s_noise
                    )
                    back_lum = np.mean(stimulus_float[0])

                _, drift_int = generate_paths_wrapper(
                    eye_movement_method,
                    parameters["T"],
                    parameters["pps"],
                    parameters["ppd"],
                    parameters["D"],
                )
                drift_int = drift_int.astype(np.float16)

                model_stimulus_gpu = cp.asarray(stimulus_float)
                drift_int_gpu = cp.asarray(drift_int)

                if edge_detection_model == "ST_M_N":
                    model_output_gpu = run_active_model_gpu(
                        model_stimulus_gpu,
                        drift_int_gpu,
                        back_lum=back_lum,
                        sfilts=dogs,
                        tfilt=tfilt,
                        rb=parameters["remove_b"],
                        integrate="mean2",
                        norm=True,
                        mempool=mempool,
                        cache=cache,
                    )
                elif edge_detection_model == "S_V_N":
                    model_output_gpu = run_active_model_gpu(
                        model_stimulus_gpu,
                        drift_int_gpu,
                        back_lum=back_lum,
                        sfilts=dogs,
                        tfilt=1.0,
                        rb=parameters["remove_b"],
                        integrate="var",
                        norm=True,
                        mempool=mempool,
                        cache=cache,
                    )
                else:
                    raise ValueError(
                        "edge_detection_model must be either 'ST_M_N' or 'S_V_N'"
                    )

                model_output_cpu = model_output_gpu.get()
                mempool.free_all_blocks()
                corr = quantify_edges(model_output_cpu, gt_template)
                correlation_scores.append(corr)

                if best_correlation is None or corr > best_correlation:
                    best_correlation = corr
                    best_output_image = model_output_cpu

            # Save the best output image
            noise_indicator = "noise" if s_noise_value > 0 else "no_noise"
            output_image_name = f"{img_name}_{noise_indicator}_D{round((parameters['D']), 3)}_pps{parameters['pps']}_{parameters['eye_movement_method']}.png"
            output_image_path = os.path.join(output_dir, output_image_name)
            fig, ax = plt.subplots()
            cax = ax.imshow(best_output_image, cmap="pink", extent=vextent_adapt)
            plt.colorbar(cax)
            plt.axis("off")  # To remove the axes around the image
            plt.savefig(
                output_image_path, bbox_inches="tight", pad_inches=0
            )  # Save the image without padding and white borders
            plt.close(fig)  # Close the figure to free up resources

            current_result = {
                "img_name": img_name,
                "D": parameters["D"],
                "pps": parameters["pps"],
                "s_noise": current_s_noise,
                "n_trials": parameters["n_trials"],
                "T": parameters["T"],
                "ppd": parameters["ppd"],
                "n_filters": parameters["n_filters"],
                "correlation_scores": correlation_scores,
                "stimulus_size": stimulus_size,
                "model": edge_detection_model,
                "eye_movement_method": parameters["eye_movement_method"],
            }

            results_list.append(current_result)
            stop = time.time()
            print("Elapsed time: " + str(stop - start) + "s")
            print("Best correlation: " + str(best_correlation))
        print("------------------------------------------------")

    cache.clear()
    drift_int_gpu = None
    tfilt = None
    mempool.free_all_blocks()

    print("--------------------FINISHED--------------------")

    pickle_file = "{}_images_{}_{}_{}.pickle".format(
        "case2",
        np.min(parameters["images_to_be_processed"]),
        np.max(parameters["images_to_be_processed"]),
        parameters["eye_movement_method"],
    )

    with open(os.path.join(results_path, pickle_file), "wb") as handle:
        pickle.dump(results_list, handle)
