# Traffic Autoencoder

## What is this?
This repository contains the code used for encoding traffic data in my Master's Thesis (which can be found here).

## What is it useful for?
The code has commands for:
* **Tune**: perform a hyperparamater grid search to look for optimal hyperparameters for the autoencoder.
* **Train**: use the data to train an autoencoder and export both the autoencoder and the encoder part as pickled Keras 
models ready to use for encoding.
* **Encode**: load a pickled encoder and use it on some data to obtained the encoded representation.


## How to run?
The repository is designed as a command line application.

First, install the package. Within a shell, cd into this directory and:

`pip install .`

There are three commands you can use: `autoencoder-evaluate`, `autoencoder-train` and `autoencoder-train`.

#### autoencoder-evaluate

```shell script
autoencoder-evaluate --config-file /path/to/config.json \
    --experiment-definition-path /path/to/definitions/directory/
```

#### autoencoder-train

```shell script
autoencoder-train --config-file /path/to/config.json \
    --model_path /path/to/model/
```

#### autoencoder-encode

```shell script
autoencoder-encode --config-file /path/to/config.json \
    --definitions-path /path/to/definitions/directory/ \
    --encode-file /path/to/encoder.pickle
```

All commands require:
 * A config file which must be in JSON format. A template can be found in the `example-config.json`
 file.
 * The path to a folder that contains one or more "definition" files, in JSON format. In `autoencoder-evaluate`, one different definition file should be 
 created for every hyperparameter combination that needs to be tested. In `autoencoder-train`, a single definition file
 defining the hyperparameters to use while training the model is required. In `autoencoder-encode`, a single definition 
 file is required, and only for describing the data preprocessing bits, since the model already exists by then.
  A template can be found in the `example-definition.json`.




