from typing import List, Union, Tuple

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

from traffic_autoencoder.schemas import ModelDefinition


def build_model(
    definition: ModelDefinition, metrics: List[str] = None, return_encoder: bool = False
) -> Union[Tuple[Model, Model], Model]:
    """
    Builds a ready to train model according to the passed definition
    :param definition: model definition tuple with the model parameters
    :param metrics: list of metrics to be reported during training and
    evaluation
    :param return_encoder: whether the encoder should be returned
    :return: the autoencoder and, optionally, the encoder part
    """

    input_layer = Input(shape=(definition.input_shape,))

    encoding = input_layer
    for size in definition.layers:
        encoding = Dense(size, activation=definition.activation)(encoding)

    encoding = Dense(definition.encoded_size, activation=definition.activation)(
        encoding
    )

    decoding = encoding
    for size in definition.layers[::-1]:  # Reverse traverse
        decoding = Dense(size, activation=definition.activation)(decoding)

    output_layer = Dense(
        definition.input_shape, activation=definition.final_activation
    )(decoding)

    optimizers = {"adam": Adam(learning_rate=definition.learning_rate)}

    wip_model = Model(input_layer, output_layer)
    encoder_model = Model(input_layer, encoding)

    wip_model.compile(
        optimizer=optimizers[definition.optimizer],
        loss="mean_squared_error",
        metrics=metrics,
    )

    final_model = wip_model

    if return_encoder:
        return final_model, encoder_model

    return final_model


def extract_encoder(autoencoder_model: Model) -> Model:
    """
    Receives an autoencoder and extracts the encoder from it.
    :param autoencoder_model: a full autoencoder
    :return: the encoder model
    """

    encoded_layer_index = int(
        len(autoencoder_model.layers) / 2
    )  # Assuming symmetrical architecture

    encoded_layers = autoencoder_model.input
    for layer in autoencoder_model.layers[2 : encoded_layer_index + 1]:
        encoded_layers = encoded_layers(layer)

    encoder = Model(autoencoder_model.input, encoded_layers)

    return encoder
