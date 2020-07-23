import pickle

from traffic_autoencoder.models import build_model
from traffic_autoencoder.schemas import ExperimentDefinition, Config
from traffic_autoencoder.tuning import load_dataset, EvaluationAttempt


def main_train(config: Config, model_definition: ExperimentDefinition):
    """
    Trains a model with certain parameters, obtains performance metrics on a
    validation set and returns actual and predicted values for that set, as
    well as a pickle file with the trained model.
    :return: None
    """

    # Load data
    train_data, validation_data = load_dataset(
        config.dataset_path, model_definition.data_preprocessing
    )

    # Train & Validate
    model, encoder = build_model(
        model_definition.model_definition, model_definition.metrics, return_encoder=True
    )
    attempt = EvaluationAttempt(
        model,
        model_definition,
        model_definition.fit_parameters,
        train_data,
        validation_data,
    )
    attempt.train_model()
    attempt.test_model()

    with open(config.results_path + "autoencoder.pickle", "wb") as output_pickle:
        pickle.dump(attempt.model, output_pickle)

    with open(config.results_path + "encoder.pickle", "wb") as output_pickle:
        pickle.dump(encoder, output_pickle)
