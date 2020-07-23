import json
import logging
import os
import random
from typing import Tuple, Union, Any, Iterable, List

import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler

from traffic_autoencoder import defaults
from traffic_autoencoder.schemas import (
    Config,
    DataPreprocessing,
    ModelDefinition,
    FitParameters,
    ExperimentDefinition,
)

logger = logging.getLogger(defaults.logger_name + "." + __name__)


class StreamToLogger(object):
    """
   Fake file-like stream object that redirects writes to a logger instance.
   """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())


def configure_logger(config: Config, verbose: bool) -> logging.Logger:
    """
    Reads the logging config and returns a ready logger. This logger name is
    hardcoded and share in all project modules.
    :param config: general script configuration
    :param verbose: whether to give all details on the process execution
    :return: a configured logger instance
    """

    logger = logging.getLogger(defaults.logger_name)
    logger.setLevel(logging.DEBUG)

    # sys.stdout = StreamToLogger(logger)
    # sys.stderr = StreamToLogger(logger, log_level=logging.ERROR)

    file_handler = logging.FileHandler(config.logging_path)
    file_level = logging.DEBUG
    file_handler.setLevel(file_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def load_dataset(
    dataset_path: str,
    preprocessing_parameters: DataPreprocessing,
    get_time_index: bool = False,
) -> Union[Tuple[numpy.array, numpy.array], Tuple[numpy.array, numpy.array, Any]]:
    """
    Terribly hardcoded loading function to load the traffic dataset. If you
    need to upload other data, by all means forget about this and build a
    function specific to your use case. Only requirement is that the output
    is completely ready for feeding straight into the model as properly
    dimensioned numpy arrays.

    :param dataset_path: the path where the file is.
    :param get_time_index: whether to include the time index in the output
    or not
    :param preprocessing_parameters: data object with the parameters for each
    specific loading
    :return: the data in the right dimensionality
    """

    raw_dataset = pandas.read_csv(dataset_path, delimiter=";")

    # Merge date and hour into a proper datetime
    raw_dataset["str_uur"] = raw_dataset["uur"].astype(str)
    raw_dataset["str_uur"] = raw_dataset["str_uur"].apply(
        lambda hour: hour if len(hour) == 2 else "0" + hour
    )
    raw_dataset["interval_start_datetime"] = pandas.to_datetime(
        raw_dataset["datum"] + " " + raw_dataset["str_uur"] + ":00:00",
        format="%d/%m/%Y %H:%M:%S",
    )

    # Merge camera name and way into a single ID
    raw_dataset["rijrichting"] = raw_dataset["rijrichting"].astype(str)

    raw_dataset["camera_id"] = (
        raw_dataset["camera_naam"] + " - way " + raw_dataset["rijrichting"]
    )

    # Remove unnecesarry fields
    raw_dataset.drop(
        [
            "datum",
            "str_uur",
            "uur",
            "camera_naam",
            "camera_kijkrichting",
            "rijrichting",
        ],
        axis=1,
        inplace=True,
    )

    # Filter only camera ID's we are interested in
    if preprocessing_parameters.camera_method == "all":
        camera_selection = defaults.CAMERAS_OF_INTEREST
    elif preprocessing_parameters.camera_method == "random":
        camera_selection = random.sample(
            defaults.CAMERAS_OF_INTEREST, k=preprocessing_parameters.camera_count
        )
    elif preprocessing_parameters.camera_method == "explicit_selection":
        camera_selection = preprocessing_parameters.camera_selection
    else:
        raise ValueError(
            f"Camera method {preprocessing_parameters.camera_method} is not a valid method"
        )

    raw_dataset = raw_dataset[raw_dataset["camera_id"].isin(camera_selection)]

    # Sort by time
    raw_dataset.sort_values(by=["interval_start_datetime"])

    # Pivot cameras as columns
    # Missing values get counted as 0
    raw_dataset = raw_dataset.pivot_table(
        index="interval_start_datetime",
        columns="camera_id",
        values="intensiteit",
        fill_value=0,
    )

    # Remove data out of time range
    raw_dataset = raw_dataset.loc[
        (raw_dataset.index > preprocessing_parameters.earliest)
        & (raw_dataset.index < preprocessing_parameters.latest)
    ]

    # Include forward and backward hours
    shaped_dataset = hour_grouper(
        raw_dataset,
        preprocessing_parameters.hours_backward,
        preprocessing_parameters.hours_forward,
    )

    # Extract time_index
    time_index = shaped_dataset.index

    # Build a scaler
    scaler = MinMaxScaler()
    scaler.fit(shaped_dataset.to_numpy())

    # Split into train and set
    train_dataset = shaped_dataset[
        shaped_dataset.index < preprocessing_parameters.split_datetime
    ]
    test_dataset = shaped_dataset[
        shaped_dataset.index >= preprocessing_parameters.split_datetime
    ]

    # Turn into numpy arrays with right dimensionality
    train_dataset = train_dataset.to_numpy()
    test_dataset = test_dataset.to_numpy()

    # Scale
    train_dataset = scaler.transform(train_dataset)
    test_dataset = scaler.transform(test_dataset)

    if get_time_index:
        return train_dataset, test_dataset, time_index
    return train_dataset, test_dataset


def hour_grouper(
    data: pandas.DataFrame, steps_backward: int = 1, steps_forward: int = 0
):
    # Conceptually, going back in time one hour is unavoidable. Thus, the first
    # hour back is removed as it is unavoidably happening.
    steps_backward -= 1
    if steps_backward == 0 and steps_forward == 0:
        return data

    camera_column_names = data.columns.values

    data["central_time"] = data.index

    for i in range(1, steps_backward + 1):
        data[f"backward_{i}"] = data["central_time"] + pandas.Timedelta(hours=i)
        data = data.merge(
            right=data[[f"backward_{i}", *camera_column_names]],
            how="left",
            left_on="central_time",
            right_on=f"backward_{i}",
            suffixes=("", f"_back{i}"),
        )
        data = data.drop([f"backward_{i}", f"backward_{i}_back{i}"], axis=1)

    for i in range(1, steps_forward + 1):
        data[f"forward_{i}"] = data["central_time"] - pandas.Timedelta(hours=i)
        data = data.merge(
            right=data[[f"forward_{i}", *camera_column_names]],
            how="left",
            left_on="central_time",
            right_on=f"forward_{i}",
            suffixes=("", f"_forw{i}"),
        )
        data = data.drop([f"forward_{i}", f"forward_{i}_forw{i}"], axis=1)

    data.index = data["central_time"]
    data = data.drop(["central_time"], axis=1)

    data = data.dropna(axis=0)

    return data


def load_definitions(
    experiment_definitions_path: str,
    sections: Iterable[Union[ModelDefinition, FitParameters, DataPreprocessing]] = (
        ModelDefinition,
        FitParameters,
        DataPreprocessing,
    ),
) -> List[ExperimentDefinition]:
    """
    Reads all the experiment definitions in the path and returns them.

    :param experiment_definitions_path: the path where definitions are stored
    as JSON files.
    :param sections: which sections of the definition should be loaded
    :return: a list of experiment definitions
    """

    definition_files_paths = os.listdir(experiment_definitions_path)

    definitions = []
    for definition_file_path in definition_files_paths:
        with open(
            os.path.join(experiment_definitions_path, definition_file_path), "r"
        ) as definition_file:
            definition_data = json.load(definition_file)

            model_definition = None
            if ModelDefinition in sections:
                model_data = definition_data["model"]
                model_definition = ModelDefinition(
                    input_shape=model_data["input_shape"],
                    layers=model_data["layers"],
                    encoded_size=model_data["encoded_size"],
                    activation=model_data["activation"],
                    final_activation=model_data["final_activation"],
                    loss_function=model_data["loss_function"],
                    optimizer=model_data["optimizer"],
                    learning_rate=model_data["learning_rate"],
                )

            fit_parameters = None
            metrics = None
            if FitParameters in sections:
                fit_parameters_data = definition_data["fit_parameters"]
                fit_parameters = FitParameters(
                    batch_size=fit_parameters_data["batch_size"],
                    epochs=fit_parameters_data["epochs"],
                )
                metrics = definition_data["metrics"]

            data_preprocessing = None
            if DataPreprocessing in sections:
                data_preprocessing_data = definition_data["data_preprocessing"]
                data_preprocessing = DataPreprocessing(
                    earliest=data_preprocessing_data["earliest"],
                    latest=data_preprocessing_data["latest"],
                    split_datetime=data_preprocessing_data["split_datetime"],
                    cv_folds=data_preprocessing_data["cv_folds"],
                    test_size=data_preprocessing_data["test_size"],
                    hours_backward=data_preprocessing_data["hours_backward"],
                    hours_forward=data_preprocessing_data["hours_forward"],
                    camera_method=data_preprocessing_data["camera_method"],
                    camera_count=data_preprocessing_data["camera_count"],
                    camera_selection=data_preprocessing_data["camera_selection"],
                )

            definition = ExperimentDefinition(
                model_definition=model_definition,
                metrics=metrics,
                fit_parameters=fit_parameters,
                data_preprocessing=data_preprocessing,
            )

            definitions.append(definition)

    return definitions
