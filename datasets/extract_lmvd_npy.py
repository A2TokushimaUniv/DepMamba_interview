import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from logzero import logger


def extract_visual_features(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".csv"):
                input_file = os.path.join(root, file)

                relative_path = os.path.relpath(input_file, input_folder)
                output_subfolder = os.path.join(
                    output_folder, os.path.dirname(relative_path)
                )
                os.makedirs(output_subfolder, exist_ok=True)
                output_file = os.path.join(
                    output_subfolder, f"{os.path.splitext(file)[0]}_visual.npy"
                )

                if not os.path.exists(output_file):
                    extract_features_single(input_file, output_file)

    logger.info(f"Features extracted successfully and saved to {output_folder}")


def extract_features_single(input_file, output_file):
    coordinates_columns = (
        ["frame"]
        + ["x_" + str(i) for i in range(68)]
        + ["y_" + str(i) for i in range(68)]
    )
    data = pd.read_csv(input_file, sep=r",\s*|\s*,\s*", engine="python")
    data = data[coordinates_columns]
    # Downsampling the data
    # Only one data point every 30 frames is used
    data = data[(data["frame"] - 1) % 30 == 0][:]

    data = data.values
    data = data[:, 1:]

    data = preprocessing.scale(data, axis=-1)
    np.save(output_file, data)


if __name__ == "__main__":
    extract_visual_features("./lmvd/Video_feature", "./lmvd/visual")