from setuptools import setup

setup(
    name="traffic_autoencoder",
    version="0.1",
    py_modules=["cli"],
    install_requires=[
        "absl-py==0.9.0",
        "astor==0.8.1",
        "cachetools==4.1.0",
        "certifi==2020.4.5.1",
        "chardet==3.0.4",
        "click==7.1.2",
        "cycler==0.10.0",
        "gast==0.2.2",
        "google-auth==1.16.1",
        "google-auth-oauthlib==0.4.1",
        "google-pasta==0.2.0",
        "grpcio==1.29.0",
        "h5py==2.10.0",
        "idna==2.9",
        "importlib-metadata==1.6.0",
        "joblib==0.15.1",
        "Keras==2.3.1",
        "Keras-Applications==1.0.8",
        "Keras-Preprocessing==1.1.2",
        "kiwisolver==1.2.0",
        "Markdown==3.2.2",
        "matplotlib==3.2.1",
        "numpy==1.18.5",
        "oauthlib==3.1.0",
        "opt-einsum==3.2.1",
        "pandas==1.0.4",
        "pkg-resources==0.0.0",
        "protobuf==3.12.2",
        "pyasn1==0.4.8",
        "pyasn1-modules==0.2.8",
        "pyparsing==2.4.7",
        "python-dateutil==2.8.1",
        "pytz==2020.1",
        "PyYAML==5.3.1",
        "requests==2.23.0",
        "requests-oauthlib==1.3.0",
        "rsa==4.0",
        "scikit-learn==0.23.1",
        "scipy==1.4.1",
        "six==1.15.0",
        "tensorboard==2.1.1",
        "tensorflow==2.1.0",
        "tensorflow-estimator==2.1.0",
        "tensorflow-gpu==2.5.1",
        "termcolor==1.1.0",
        "threadpoolctl==2.1.0",
        "urllib3==1.25.9",
        "Werkzeug==1.0.1",
        "wrapt==1.12.1",
        "zipp==3.1.0",
    ],
    entry_points="""
        [console_scripts]
        autoencoder-evaluate=traffic_autoencoder.cli:evaluate
        autoencoder-train=traffic_autoencoder.cli:train
        autoencoder-encode=traffic_autoencoder.cli:encode
    """,
)
