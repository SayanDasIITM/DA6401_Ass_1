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

## Code Organization
The code is organized into modular components to ensure clarity, maintainability, and scalability:

- `Data Loading & Preprocessing`:
Functions are provided to load datasets (Fashion-MNIST or MNIST), perform train/validation/test splits, scale the data, reshape images, and one-hot encode labels.

- `Activation Functions & Derivatives`:
Contains implementations for various activation functions (ReLU, Sigmoid, Tanh, Softmax) along with their derivatives, which are essential for the backpropagation process.

- `Weight Initialization`:
Supports different methods (Random, Xavier) to initialize network weights and biases, ensuring that the network starts training with appropriate scale.

- `Neural Network Model`:
The NeuralNetwork class encapsulates both forward and backward propagation. It supports multiple activation functions and is configurable via command-line arguments.

- `Loss Functions & Accuracy`:
Implements commonly used loss functions (cross-entropy, mean squared error) and an accuracy metric to evaluate model performance.

- `Optimizer with Gradient Clipping`:
The Optimizer class includes various optimization algorithms (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam) and uses gradient clipping to prevent exploding gradients.

- `Trainer Class`:
Manages the training loop, including batch processing, metric logging to wandb, and evaluation on validation and test sets.

- `Analysis & Visualization`:
Provides tools for generating interactive plots such as confusion matrices and hyperparameter analysis plots (parallel coordinates, correlation heatmaps).

This structure facilitates easier debugging, updates, and scalability of the project.

## Report

The detailed experiment report is available at:
[Weights & Biases Report](https://api.wandb.ai/links/cs24m044-iit-madras-alumni-association/vzk6r03m)


## Self Declaration
I, Sayan Das (cs24m044), declare that I have implemented the code and report independently, without unauthorized collaboration.

## Contact
For any queries, open an issue in this repository.

