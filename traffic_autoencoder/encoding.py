import pickle
from typing import IO

import numpy
import pandas
from keras import Model

from traffic_autoencoder.schemas import Config, DataPreprocessing
from traffic_autoencoder.utils import load_dataset


def main_encoder(
    config: Config, encoder: Model, data_preprocessing_params: DataPreprocessing
) -> None:
    # Get the data
    train_data, test_data, time_index = load_dataset(
        config.dataset_path, data_preprocessing_params, get_time_index=True
    )

    # Unite both arrays
    all_data = numpy.concatenate((train_data, test_data))

    # Run through the encoder
    encoded_data = encoder.predict(all_data)

    # Turn into dataframe with time index
    encoded_data = pandas.DataFrame(encoded_data, index=time_index)

    # Store output
    encoded_data.to_csv(config.results_path + "encoded_features.csv", sep=",")


def load_encoder(encoder_file: IO) -> Model:
    """
    Reads the encoder pickle file and unpickles it.
    :param encoder_file: the IO object to the file
    :return: the unpickled keras model
    """
    return pickle.load(encoder_file)
