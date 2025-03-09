# Fashion-MNIST Neural Network Training

This repository contains Python scripts for loading, visualizing, and training a neural network on the Fashion-MNIST dataset using `wandb` for logging.

## Installation

Before running the scripts, install the required dependencies using:

```bash
pip install numpy matplotlib seaborn plotly scikit-learn keras pandas wandb
```

## Usage

### 1. Visualizing Fashion-MNIST Samples

Run the `Question_1.py` script to visualize sample images from the dataset and log them to Weights & Biases.

```bash
python Question_1.py
```

### 2. Training the Neural Network

The `train.py` script trains a fully connected neural network with configurable hyperparameters.

#### Running the default training:
```bash
python train.py
```

#### Available Arguments:

- `--dataset` : Dataset to use (`fashion_mnist` or `mnist`). Default: `fashion_mnist`
- `--epochs` : Number of training epochs. Default: `10`
- `--batch_size` : Batch size. Default: `64`
- `--loss` : Loss function (`mean_squared_error` or `cross_entropy`). Default: `cross_entropy`
- `--optimizer` : Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`). Default: `adam`
- `--learning_rate` : Learning rate. Default: `0.001`
- `--num_layers` : Number of hidden layers. Default: `3`
- `--hidden_size` : Neurons per hidden layer. Default: `128`
- `--activation` : Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`). Default: `ReLU`

#### Example Usage:
```bash
python train.py --dataset mnist --epochs 20 --batch_size 32 --optimizer adam --learning_rate 0.0005
```

### 3. Running Hyperparameter Analysis

To analyze hyperparameter performance:
```bash
python train.py --analyze
```

### 4. Running a WandB Sweep for Hyperparameter Tuning

To start a sweep:
```bash
python train.py --sweep
```

## Logging with WandB

Ensure you are logged into WandB before running the scripts. If required, create a `wandb_api.txt` file with your API key.
```bash
wandb login
```

## Results

- Training and validation loss/accuracy are logged to WandB.
- A confusion matrix is generated after training.
- Hyperparameter analysis is stored in `parallel_coordinates.html` and `correlation_heatmap.png`.

## License

This project is open-source under the MIT License.

