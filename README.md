# Federated Learning Simulation For Healthcare records-based data

This repository contains a Federated Learning simulation framework that simulates clients' training locally and sends their model updates to a central server for aggregation. The framework is based on `Hydra` and uses custom modules for dataset preparation, client-side training, and server-side aggregation.

## Features
- **Client-Server Architecture:** Simulates federated learning using clients and a central server.
- **Customizable Configurations:** Uses `Hydra` for easy configuration of various hyperparameters and settings.
- **Federated Averaging (FedAvg):** Implements the standard FedAvg algorithm for model aggregation.
- **Data Preparation:** Includes functionality to split and prepare datasets for clients.
- **Performance Tracking:** Logs accuracy and loss after each training round.

## Getting Started

### Prerequisites (you have to install the requirement.txt)
- Python 3.7+
- `hydra-core`
- `omegaconf`

# Data Set :
download the dataset from this Kaggle repo. "https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data"
This is the file name: heart_2022_no_nans.csv

# running process:
 - Download the repository and the requirement.txt file.
 - go to the conf file and change the yamel file. [the dataset path, model_type_one_cluster, target_clt, etc.]
 - then run the main file as is.
 - you will see, the evaluation after each round.

You can install the required dependencies using the following command:

pip install -r requirement.txt




##Configuration File
The parameters for the training process are defined in a YAML file located in the conf directory. The file used is named base.yaml
you need to define the parameters in this file to make the running easier. 
