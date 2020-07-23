import json
from typing import IO

from traffic_autoencoder.schemas import Config


def load_config(config_file: IO) -> Config:
    """
    Reads the config file and returns the config object.
    :param config_file: file object with the config
    :return: the config parameters
    """
    config_data = json.load(config_file)

    config = Config(
        logging_path=config_data.get("logging_path"),
        dataset_path=config_data.get("dataset_path"),
        results_path=config_data.get("results_path"),
    )

    return config
