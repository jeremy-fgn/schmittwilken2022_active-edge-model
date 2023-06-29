"""
File to transfer the results from main_case2.gpu.py to a csv file.

Last update on 29.06.2023
@author: jeremy-fgn
"""

import os
import pickle
import glob

import pandas as pd

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
results_path = os.path.join(project_root, "data")


if __name__ == "__main__":
    pickle_files = glob.glob(os.path.join(results_path, "case2_*" + ".pickle"))

    raw_results = []

    # Load the contents of each pickle file and extend the 'results' list
    for pickle_file in pickle_files:
        with open(pickle_file, "rb") as handle:
            current_results = pickle.load(handle)
            raw_results.extend(current_results)

    # Create a pandas dataframe from the list of dictionaries
    results = pd.DataFrame(
        columns=[
            "eye_movement_method",
            "D",
            "pps",
            "s_noise",
            "img_name",
            "mean_correlation_score",
            "model",
        ]
    )

    # Loop through each dictionary in the list and extract the required information
    for current_result in raw_results:
        eye_movement_method = current_result["eye_movement_method"]
        edge_detection_model = current_result["model"]
        D = current_result["D"]
        pps = current_result["pps"]
        s_noise = current_result["s_noise"]
        img_name = current_result["img_name"]
        correlation_scores = current_result["correlation_scores"]
        mean_correlation_score = sum(correlation_scores) / len(correlation_scores)

        # Append a row to the results dataframe with the extracted information
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "eye_movement_method": [eye_movement_method],
                        "model": [edge_detection_model],
                        "D": [D],
                        "pps": [pps],
                        "s_noise": [s_noise],
                        "img_name": [img_name],
                        "mean_correlation_score": [mean_correlation_score],
                    }
                ),
            ],
            ignore_index=True,
        )

    # Save the results dataframe to a csv file
    results.to_csv(os.path.join(results_path, "results.csv"), index=False)
