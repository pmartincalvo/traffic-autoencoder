import collections


Config = collections.namedtuple(
    "Config", ["logging_path", "dataset_path", "results_path"]
)

ModelDefinition = collections.namedtuple(
    "ModelDefinition",
    [
        "input_shape",
        "layers",
        "encoded_size",
        "activation",
        "final_activation",
        "loss_function",
        "optimizer",
        "learning_rate",
    ],
)
FitParameters = collections.namedtuple("FitParameters", ["batch_size", "epochs"])
DataPreprocessing = collections.namedtuple(
    "DataPreprocessing",
    [
        "earliest",
        "latest",
        "split_datetime",
        "cv_folds",
        "test_size",
        "hours_backward",
        "hours_forward",
        "camera_method",
        "camera_count",
        "camera_selection",
    ],
)
ExperimentDefinition = collections.namedtuple(
    "ExperimentDefinition",
    ["model_definition", "metrics", "fit_parameters", "data_preprocessing"],
)

EvaluationAttemptResults = collections.namedtuple(
    "EvaluationAttempt", ["experiment_definition", "metric_results"]
)
