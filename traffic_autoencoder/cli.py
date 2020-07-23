import click

from traffic_autoencoder.config import load_config
from traffic_autoencoder.encoding import main_encoder, load_encoder
from traffic_autoencoder.schemas import DataPreprocessing
from traffic_autoencoder.training import main_train
from traffic_autoencoder.tuning import main_evaluate
from traffic_autoencoder.utils import configure_logger, load_definitions


@click.option("config_file", "--config-file", type=click.File())
@click.option(
    "experiment_definitions_path", "--experiment-definitions-path", type=click.Path()
)
@click.option("--verbose", is_flag=True)
@click.command()
def evaluate(config_file, experiment_definitions_path, verbose):
    config = load_config(config_file)
    logger = configure_logger(config, verbose)
    experiment_definitions = load_definitions(experiment_definitions_path)

    logger.info("Starting evaluation")
    main_evaluate(config, experiment_definitions)
    logger.info("Finished execution")


@click.option("model_path", "--model-path", type=click.Path())
@click.option("config_file", "--config-file", type=click.File())
@click.option("--verbose", is_flag=True)
@click.command()
def train(model_path, config_file, verbose):
    config = load_config(config_file)
    logger = configure_logger(config, verbose)
    model_definition = load_definitions(model_path)[0]

    logger.info("Starting training")
    main_train(config, model_definition)
    logger.info("Finished training")


@click.option("config_file", "--config-file", type=click.File())
@click.option("encoder_file", "--encode-file", type=click.File(mode="rb"))
@click.option("definitions_path", "--definitions-path", type=click.Path())
@click.option("--verbose", is_flag=True)
@click.command()
def encode(config_file, definitions_path, encoder_file, verbose):
    config = load_config(config_file)
    logger = configure_logger(config, verbose)
    preprocessing_params = load_definitions(
        definitions_path, sections=[DataPreprocessing]
    )[0].data_preprocessing
    encoder = load_encoder(encoder_file)

    logger.info("Start encoding")
    main_encoder(config, encoder, data_preprocessing_params=preprocessing_params)
    logger.info("Finished encoding")
