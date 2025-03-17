# DA6401 - Assignment 1: Neural Network Training and Experiment Tracking

This repository contains code for DA6401 - Assignment 1, which involves implementing a feedforward neural network with backpropagation and tracking experiments using Weights & Biases (wandb). The github_link for the following is :- https://github.com/SayanDasIITM/DA6401_Ass_1/tree/main

## Contents
- `Question_1.py`: Loads and visualizes the Fashion-MNIST dataset.
- `train.py`: Implements and trains a neural network using different optimizers, activation functions, and hyperparameters, with wandb logging.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib seaborn pandas plotly scikit-learn keras tensorflow wandb
```

## How to Run

### Important Note on Running Sweeps
After running a sweep, a file `sweep_id.txt` is created to store the sweep ID. If you want to run a new sweep, you must first delete the existing `sweep_id.txt` file before starting a new sweep.


### Running `Question_1.py`

This script loads and visualizes sample images from the Fashion-MNIST dataset and logs them to Weights & Biases.

```bash
python Question_1.py
```

### Running `train.py`

This script trains a feedforward neural network with various configurable options.

Basic command:

```bash
python train.py
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-wp, --wandb_project` | "neural-network-sweep" | Weights & Biases project name |
| `-we, --wandb_entity` | "cs24m044-iit-madras-alumni-association" | Weights & Biases entity name |
| `-d, --dataset` | "fashion_mnist" | Dataset to use (`mnist` or `fashion_mnist`) |
| `-e, --epochs` | 10 | Number of training epochs |
| `-b, --batch_size` | 64 | Batch size |
| `-l, --loss` | "cross_entropy" | Loss function (`cross_entropy` or `mean_squared_error`) |
| `-o, --optimizer` | "adam" | Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`) |
| `-lr, --learning_rate` | 0.001 | Learning rate |
| `-m, --momentum` | 0.9 | Momentum for optimizers |
| `-nhl, --num_layers` | 3 | Number of hidden layers |
| `-sz, --hidden_size` | 128 | Neurons per hidden layer |
| `-a, --activation` | "ReLU" | Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`) |
| `--compare_losses` | False | Compare cross-entropy vs. MSE loss |
| `--mnist_recommend` | False | Run recommended MNIST configurations |
| `--sweep` | False | Run wandb sweep experiment |
| `--analyze` | False | Run hyperparameter analysis |

Example:

```bash
python train.py -e 20 -b 32 -o rmsprop -lr 0.0005 -nhl 4 -sz 256
```

## Report

The detailed experiment report is available at:
[Weights & Biases Report](https://api.wandb.ai/links/cs24m044-iit-madras-alumni-association/vzk6r03m)


## Self Declaration
I, Sayan Das (cs24m044), declare that I have implemented the code and report independently, without unauthorized collaboration.

## Contact
For any queries, open an issue in this repository.

