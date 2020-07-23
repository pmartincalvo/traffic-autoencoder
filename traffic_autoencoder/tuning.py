import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple, List

import numpy
import pandas
from keras import Model
from sklearn.model_selection import train_test_split

from traffic_autoencoder import defaults
from traffic_autoencoder.models import build_model
from traffic_autoencoder.schemas import (
    EvaluationAttemptResults,
    FitParameters,
    ExperimentDefinition,
)
from traffic_autoencoder.utils import load_dataset

logger = logging.getLogger(defaults.logger_name + "." + __name__)


def main_evaluate(
    config: NamedTuple, experiment_definitions: List[ExperimentDefinition]
) -> None:
    """
    Main function for autoencoder setup evaluations. Loads parameters, trains
    and measures results of different setups, storing the results of each so
    that a final model can be chosen.
    :param config: general script configuration
    :param experiment_definitions: all the different autoencoder setups to
    be tested.
    :return: None
    """
    logger.info(
        f"Beginning experiments. Total number of experiments: {len(experiment_definitions)}"
    )
    last_data_preprocessing = None
    evaluation_session = EvaluationSession()
    attempt_batch = []
    for counter, experiment_definition in enumerate(experiment_definitions):
        logger.info(f"Loading dataset from {config.dataset_path}")
        if last_data_preprocessing == experiment_definition.data_preprocessing:
            logger.info(f"Same preprocessing parameters as last one. Can reuse.")
        else:
            all_train_data, validation_data = load_dataset(
                config.dataset_path, experiment_definition.data_preprocessing
            )
            last_data_preprocessing = experiment_definition.data_preprocessing
            logger.info(f"Data loaded and split")

        logger.info(
            f"Preparing {experiment_definition.data_preprocessing.cv_folds} folds for cross-validation"
        )
        for fold in range(0, experiment_definition.data_preprocessing.cv_folds):
            train_data, test_data = train_test_split(
                all_train_data,
                test_size=experiment_definition.data_preprocessing.test_size,
                shuffle=True,
                random_state=(random.randint(0, 100)),
            )
            model = build_model(
                experiment_definition.model_definition, experiment_definition.metrics
            )
            logger.info("Model built successfully")
            attempt = EvaluationAttempt(
                model,
                experiment_definition,
                experiment_definition.fit_parameters,
                train_data,
                test_data,
            )

            if not config.multiprocessing_cores:
                logger.info("Starting training")
                attempt.train_model()
                logger.info("Finished training")
                evaluation_session.record_new_attempt(attempt)
            else:
                attempt_batch.append(attempt)
                if len(attempt_batch) == config.multiprocessing_cores:
                    logging.info("Starting parallel batch")
                    with ProcessPoolExecutor(
                        max_workers=config.multiprocessing_cores
                    ) as executor:
                        results = executor.map(
                            EvaluationAttempt.train_and_return, attempt_batch
                        )
                    for result in results:
                        evaluation_session.record_new_results(result)
                    attempt_batch = []
                    logging.info("Finished batch")

        logger.info(f"Experiment {counter}/{len(experiment_definitions)} done")

    # Collect all results
    logger.info("Finishing session")
    results = evaluation_session.finish_session()
    pandas.DataFrame(results).to_csv(os.path.join(config.results_path, "results.csv"))


class EvaluationAttempt:
    """
    Encapsulates a configured autoencoder model, together with training and
    testing data. Once the model is trained and tested, data on the final
    results is available.
    """

    def __init__(
        self,
        model: Model,
        experiment_definition: ExperimentDefinition,
        fit_parameters: FitParameters,
        train_data: numpy.array,
        test_data: numpy.array,
    ):
        """
        Receives the model and data.
        :param model: a ready to train autoencoder model
        :param train_data: data for training, already formatted for feeding
        into the model
        :param test_data: data for testing, already formatted for feeding
        into the model
        """
        self.model = model
        self.experiment_definition = experiment_definition
        self.fit_parameters = fit_parameters
        self.train_data = train_data
        self.test_data = test_data

    def train_model(self):
        """
        Executes the training of the model and stores the trained model.
        :return: None
        """
        self.history = self.model.fit(
            x=self.train_data,
            y=self.train_data,
            batch_size=self.fit_parameters.batch_size,
            epochs=self.fit_parameters.epochs,
            validation_data=(self.test_data, self.test_data),
        )
        return self

    def test_model(self):
        """
        Uses the trained model to process the test data and keeps quality
        metrics.
        :return: None
        """
        self.predicted_values = self.model.predict(
            self.test_data, batch_size=self.fit_parameters.batch_size
        )

    def get_results(self) -> EvaluationAttemptResults:
        """
        Returns all the stored details on the attempt.
        :return:
        """
        metric_results = {key: value[-1] for key, value in self.history.history.items()}

        return EvaluationAttemptResults(
            experiment_definition=self.experiment_definition,
            metric_results=metric_results,
        )

    @staticmethod
    def train_and_return(some_attempt) -> EvaluationAttemptResults:
        """
        Combines training and returning results to better suit parallel
        processing
        :return:
        """
        some_attempt.train_model()
        return some_attempt.get_results()


class EvaluationSession:
    """
    Stores results from several experiments and groups all of them together.
    """

    def __init__(self):
        self._results = []

    def record_new_attempt(self, new_attempt: EvaluationAttempt):
        self._results.append(new_attempt.get_results())

    def record_new_results(self, new_results: EvaluationAttemptResults):
        self._results.append(new_results)

    def finish_session(self):
        nice_format_results = []

        for result in self._results:
            nice_result = {
                "batch_size": result.experiment_definition.fit_parameters.batch_size,
                "epochs": result.experiment_definition.fit_parameters.epochs,
                "activation": result.experiment_definition.model_definition.activation,
                "encoded_size": result.experiment_definition.model_definition.encoded_size,
                "layers": result.experiment_definition.model_definition.layers,
                "optimizer": result.experiment_definition.model_definition.optimizer,
                "learning_rate": result.experiment_definition.model_definition.learning_rate,
            }

            for metric, value in result.metric_results.items():
                nice_result[metric] = value

            nice_format_results.append(nice_result)

        return nice_format_results
