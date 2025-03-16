#!/usr/bin/env python
import os
import argparse
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import wandb
import pandas as pd

# Set environment variables for wandb.
os.environ["WANDB_TELEMETRY_DISABLED"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_PROJECT"] = "neural-network-sweep"
os.environ["WANDB_ENTITY"] = "cs24m044-iit-madras-alumni-association"
# Uncomment below for offline mode if needed.
# os.environ["WANDB_MODE"] = "offline"

# Read wandb API key from file if it exists
if os.path.exists("wandb_api.txt"):
    with open("wandb_api.txt", "r") as key_file:
        api_key = key_file.read().strip()
    os.environ["WANDB_API_KEY"] = api_key

#############################
# Data Loading & Preprocessing
#############################
def load_data_80_10_10(dataset_name):
    """
    Loads the standard Fashion-MNIST or MNIST dataset and creates an 80/10/10 split
    for training, validation, and test sets respectively.
    """
    try:
        if dataset_name.lower() == "mnist":
            from keras.datasets import mnist
            (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
        else:
            from keras.datasets import fashion_mnist
            (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    except Exception as e:
        print("Error loading dataset:", e)
        raise

    # Combine full dataset: 60K + 10K = 70K images
    X = np.concatenate([x_train_full, x_test], axis=0)
    y = np.concatenate([y_train_full, y_test], axis=0)

    # First, split 10% for test (~7K images)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )
    # Then split remaining 90% into train (80% total) and val (10% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42, shuffle=True
    )

    # Scale to [0,1] and reshape from (28,28) -> (784,)
    X_train = X_train.astype("float32") / 255.0
    X_val   = X_val.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0

    X_train = X_train.reshape((-1, 28*28))
    X_val   = X_val.reshape((-1, 28*28))
    X_test  = X_test.reshape((-1, 28*28))

    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_val   = encoder.transform(y_val.reshape(-1, 1))
    y_test  = encoder.transform(y_test.reshape(-1, 1))
    return X_train, y_train, X_val, y_val, X_test, y_test

#############################
# Activation Functions & Derivatives
#############################
class ActivationFunctions:
    @staticmethod
    def identity(x):
        return x
    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -200, 200)
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class DifferentialFunctions:
    @staticmethod
    def identity_derivative(a):
        return np.ones_like(a)
    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)
    @staticmethod
    def relu_derivative(a):
        return (a > 0).astype(float)
    @staticmethod
    def tanh_derivative(a):
        return 1 - np.power(a, 2)

#############################
# Weight Initialization
#############################
class WeightInitializer:
    @staticmethod
    def random_init(layer_sizes):
        weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 
                   for i in range(len(layer_sizes)-1)]
        biases  = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        return weights, biases

    @staticmethod
    def xavier_init(layer_sizes):
        weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
                   for i in range(len(layer_sizes)-1)]
        biases  = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        return weights, biases

#############################
# Neural Network Model
#############################
class NeuralNetwork:
    def __init__(self, layer_sizes, activation="relu", weight_init="xavier", weight_decay=0.0, output_activation="softmax"):
        self.layer_sizes = layer_sizes
        self.activation = activation.lower()
        self.weight_decay = weight_decay
        self.output_activation = output_activation.lower()
        try:
            if weight_init.lower() == "random":
                self.weights, self.biases = WeightInitializer.random_init(layer_sizes)
            else:
                self.weights, self.biases = WeightInitializer.xavier_init(layer_sizes)
        except Exception as e:
            print("Error in weight initialization:", e)
            raise
    
    def forward_pass(self, X):
        activations = [X]
        try:
            for i in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                if i == len(self.weights)-1:
                    if self.output_activation == "softmax":
                        a = ActivationFunctions.softmax(z)
                    elif self.output_activation == "identity":
                        a = ActivationFunctions.identity(z)
                    else:
                        raise ValueError("Unsupported output activation")
                else:
                    if self.activation == "sigmoid":
                        a = ActivationFunctions.sigmoid(z)
                    elif self.activation == "relu":
                        a = ActivationFunctions.relu(z)
                    elif self.activation == "tanh":
                        a = ActivationFunctions.tanh(z)
                    elif self.activation == "identity":
                        a = ActivationFunctions.identity(z)
                    else:
                        raise ValueError("Unsupported activation function")
                activations.append(a)
        except Exception as e:
            print("Error during forward pass:", e)
            raise
        return activations
    
    def backward_pass(self, activations, y_true):
        try:
            del_w = [np.zeros_like(w) for w in self.weights]
            del_b = [np.zeros_like(b) for b in self.biases]
            if self.output_activation == "identity":
                delta = 2 * (activations[-1] - y_true)
            else:
                delta = activations[-1] - y_true
            for i in reversed(range(len(self.weights))):
                del_w[i] = np.dot(activations[i].T, delta) / activations[i].shape[0]
                del_b[i] = np.sum(delta, axis=0, keepdims=True) / activations[i].shape[0]
                if i > 0:
                    delta = np.dot(delta, self.weights[i].T)
                    if self.activation == "sigmoid":
                        delta *= DifferentialFunctions.sigmoid_derivative(activations[i])
                    elif self.activation == "relu":
                        delta *= DifferentialFunctions.relu_derivative(activations[i])
                    elif self.activation == "tanh":
                        delta *= DifferentialFunctions.tanh_derivative(activations[i])
                    elif self.activation == "identity":
                        delta *= DifferentialFunctions.identity_derivative(activations[i])
        except Exception as e:
            print("Error during backward pass:", e)
            raise
        return del_w, del_b

#############################
# Loss Functions & Accuracy
#############################
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

#############################
# Optimizer with Gradient Clipping
#############################
class Optimizer:
    def __init__(self, optimizer, lr=0.001, momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, clip_value=5.0):
        self.optimizer = "nesterov" if optimizer.lower() == "nag" else optimizer.lower()
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clip_value = clip_value
        self.t = 0
        self.v_w = None
        self.v_b = None
        self.m_w = None
        self.vw_adam = None
        self.m_b = None
        self.vb_adam = None
        self.cache_w = None
        self.cache_b = None
        
    def initialize(self, weights, biases):
        self.v_w = [np.zeros_like(w) for w in weights]
        self.v_b = [np.zeros_like(b) for b in biases]
        self.m_w = [np.zeros_like(w) for w in weights]
        self.vw_adam = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.vb_adam = [np.zeros_like(b) for b in biases]
        self.cache_w = [np.zeros_like(w) for w in weights]
        self.cache_b = [np.zeros_like(b) for b in biases]
        
    def update(self, weights, biases, del_w, del_b):
        try:
            if self.v_w is None:
                self.initialize(weights, biases)
            for i in range(len(weights)):
                del_w[i] += self.weight_decay * weights[i]
                grad_norm = np.linalg.norm(del_w[i])
                if grad_norm > self.clip_value:
                    del_w[i] = del_w[i] * (self.clip_value / grad_norm)
            self.t += 1
            if self.optimizer == "sgd":
                for i in range(len(weights)):
                    weights[i] -= self.lr * del_w[i]
                    biases[i]  -= self.lr * del_b[i]
            elif self.optimizer == "momentum":
                for i in range(len(weights)):
                    self.v_w[i] = self.momentum * self.v_w[i] + self.lr * del_w[i]
                    self.v_b[i] = self.momentum * self.v_b[i] + self.lr * del_b[i]
                    weights[i] -= self.v_w[i]
                    biases[i]  -= self.v_b[i]
            elif self.optimizer == "nesterov":
                for i in range(len(weights)):
                    v_prev_w = self.v_w[i].copy()
                    v_prev_b = self.v_b[i].copy()
                    self.v_w[i] = self.momentum * self.v_w[i] - self.lr * del_w[i]
                    self.v_b[i] = self.momentum * self.v_b[i] - self.lr * del_b[i]
                    weights[i] += -self.momentum * v_prev_w + (1 + self.momentum) * self.v_w[i]
                    biases[i]  += -self.momentum * v_prev_b + (1 + self.momentum) * self.v_b[i]
            elif self.optimizer == "rmsprop":
                for i in range(len(weights)):
                    self.cache_w[i] = self.beta * self.cache_w[i] + (1 - self.beta) * (del_w[i]**2)
                    self.cache_b[i] = self.beta * self.cache_b[i] + (1 - self.beta) * (del_b[i]**2)
                    weights[i] -= self.lr * del_w[i] / (np.sqrt(self.cache_w[i]) + self.epsilon)
                    biases[i]  -= self.lr * del_b[i] / (np.sqrt(self.cache_b[i]) + self.epsilon)
            elif self.optimizer == "adam":
                for i in range(len(weights)):
                    self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * del_w[i]
                    self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * del_b[i]
                    self.vw_adam[i] = self.beta2 * self.vw_adam[i] + (1 - self.beta2) * (del_w[i]**2)
                    self.vb_adam[i] = self.beta2 * self.vb_adam[i] + (1 - self.beta2) * (del_b[i]**2)
                    m_hat_w = self.m_w[i] / (1 - self.beta1**self.t)
                    m_hat_b = self.m_b[i] / (1 - self.beta1**self.t)
                    v_hat_w = self.vw_adam[i] / (1 - self.beta2**self.t)
                    v_hat_b = self.vb_adam[i] / (1 - self.beta2**self.t)
                    weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                    biases[i]  -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            elif self.optimizer == "nadam":
                for i in range(len(weights)):
                    self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * del_w[i]
                    self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * del_b[i]
                    self.vw_adam[i] = self.beta2 * self.vw_adam[i] + (1 - self.beta2) * (del_w[i]**2)
                    self.vb_adam[i] = self.beta2 * self.vb_adam[i] + (1 - self.beta2) * (del_b[i]**2)
                    m_hat_w = self.m_w[i] / (1 - self.beta1**self.t)
                    m_hat_b = self.m_b[i] / (1 - self.beta1**self.t)
                    v_hat_w = self.vw_adam[i] / (1 - self.beta2**self.t)
                    v_hat_b = self.vb_adam[i] / (1 - self.beta2**self.t)
                    weights[i] -= self.lr * ((self.beta1 * m_hat_w + (1 - self.beta1) * del_w[i] / (1 - self.beta1**self.t)) / (np.sqrt(v_hat_w) + self.epsilon))
                    biases[i]  -= self.lr * ((self.beta1 * m_hat_b + (1 - self.beta1) * del_b[i] / (1 - self.beta1**self.t)) / (np.sqrt(v_hat_b) + self.epsilon))
            else:
                raise ValueError("Unsupported optimizer type.")
        except Exception as e:
            print("Error during optimizer update:", e)
            wandb.log({"optimizer_error": str(e)})
            raise

#############################
# Trainer Class
#############################
class Trainer:
    def __init__(self, model, optimizer, config, loss_type, x_train, y_train, x_val, y_val, x_test=None, y_test=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_type = loss_type
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        # Optionally pass test set for logging test accuracy per epoch
        self.x_test = x_test
        self.y_test = y_test
        self.test_acc_history = []  # to store test accuracy per epoch
    
    def train(self):
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        num_samples = self.x_train.shape[0]
        
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}")
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_train_shuffled = self.x_train[indices]
            y_train_shuffled = self.y_train[indices]
            batch_errors = 0
            train_losses = []
            train_accuracies = []
            
            # Training in batches
            for i in range(0, num_samples, batch_size):
                try:
                    X_batch = x_train_shuffled[i:i+batch_size]
                    y_batch = y_train_shuffled[i:i+batch_size]
                    activations = self.model.forward_pass(X_batch)
                    if np.any(np.isnan(activations[-1])):
                        raise ValueError("NaN detected in network output during forward pass.")
                    if self.loss_type == "cross_entropy":
                        loss_batch = cross_entropy_loss(y_batch, activations[-1])
                    else:
                        loss_batch = mean_squared_error(y_batch, activations[-1])
                    train_losses.append(loss_batch)
                    acc_batch = accuracy(y_batch, activations[-1])
                    train_accuracies.append(acc_batch)
                    del_w, del_b = self.model.backward_pass(activations, y_batch)
                    self.optimizer.update(self.model.weights, self.model.biases, del_w, del_b)
                except Exception as batch_error:
                    print(f"Error in batch starting at index {i}: {batch_error}")
                    wandb.log({"batch_error": str(batch_error)})
                    batch_errors += 1
                    continue
            
            # Compute full training metrics (using entire training set)
            train_activations = self.model.forward_pass(self.x_train)
            if self.loss_type == "cross_entropy":
                train_loss = cross_entropy_loss(self.y_train, train_activations[-1])
            else:
                train_loss = mean_squared_error(self.y_train, train_activations[-1])
            train_acc = accuracy(self.y_train, train_activations[-1])
            
            # Validation metrics
            activations_val = self.model.forward_pass(self.x_val)
            if self.loss_type == "cross_entropy":
                val_loss = cross_entropy_loss(self.y_val, activations_val[-1])
            else:
                val_loss = mean_squared_error(self.y_val, activations_val[-1])
            val_acc = accuracy(self.y_val, activations_val[-1])
            
            # Log training and validation metrics
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}, Batch Errors = {batch_errors}")
            
            # If test set is provided, compute test accuracy per epoch and log it
            if self.x_test is not None and self.y_test is not None:
                activations_test = self.model.forward_pass(self.x_test)
                test_acc = accuracy(self.y_test, activations_test[-1])
                self.test_acc_history.append(test_acc)
                wandb.log({"test_accuracy": test_acc, "epoch": epoch+1})
        
        wandb.run.summary["final_val_loss"] = val_loss
        wandb.run.summary["final_val_accuracy"] = val_acc
        
        # Plot test accuracy vs. epoch if test set was provided
        if self.test_acc_history:
            epochs_range = range(1, epochs+1)
            plt.figure(figsize=(6,4))
            plt.plot(epochs_range, self.test_acc_history, marker="o", linestyle="-")
            plt.xlabel("Epoch")
            plt.ylabel("Test Accuracy")
            plt.title("Test Accuracy vs. Epoch")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("test_accuracy_vs_epoch.png")
            wandb.log({"test_accuracy_vs_epoch": wandb.Image("test_accuracy_vs_epoch.png")})
    
    def evaluate(self, epoch):
        try:
            activations = self.model.forward_pass(self.x_val)
            if self.loss_type=="cross_entropy":
                loss = cross_entropy_loss(self.y_val, activations[-1])
            else:
                loss = mean_squared_error(self.y_val, activations[-1])
            acc = accuracy(self.y_val, activations[-1])
            wandb.log({"epoch": epoch+1, "epoch_loss": loss, "epoch_accuracy": acc})
            print(f"Epoch {epoch+1}: Val Loss = {loss:.4f}, Val Accuracy = {acc:.4f}")
        except Exception as eval_error:
            print(f"Error during evaluation at epoch {epoch+1}: {eval_error}")
            wandb.log({"eval_error": str(eval_error)})

#############################
# Test and Plot Confusion Matrix (Interactive)
#############################
def test_and_plot(model, loss_type, x_test, y_test):
    try:
        activations = model.forward_pass(x_test)
        if loss_type=="cross_entropy":
            test_loss = cross_entropy_loss(y_test, activations[-1])
        else:
            test_loss = mean_squared_error(y_test, activations[-1])
        test_acc = accuracy(y_test, activations[-1])
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Log final test accuracy
        wandb.run.summary["accuracy"] = test_acc

        # Prepare confusion matrix
        y_pred = np.argmax(activations[-1], axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred)

        # Normalize confusion matrix (for percentages)
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        # Define Fashion MNIST class labels
        fashion_labels = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
        ]

        # Create interactive Plotly heatmap with adjusted dimensions
        fig = px.imshow(
            cm_normalized,
            labels=dict(x="Predicted", y="True", color="Percentage"),
            x=fashion_labels,
            y=fashion_labels,
            color_continuous_scale="Blues",
            text_auto=".1%",
            aspect="auto"
        )
        fig.update_layout(width=600, height=500, margin=dict(l=50, r=50, t=50, b=50),
                          font=dict(size=10))
        
        # Build custom hover text to show raw counts and percentages
        customdata = [[cm[i, j]] for i in range(cm.shape[0]) for j in range(cm.shape[1])]
        fig.update_traces(
            customdata=np.array(customdata).reshape(cm.shape[0], cm.shape[1]),
            hovertemplate=(
                "<b>True:</b> %{y}<br>" +
                "<b>Predicted:</b> %{x}<br>" +
                "<b>Count:</b> %{customdata[0]}<br>" +
                "<b>Percentage:</b> %{z:.2%}"
            )
        )

        fig.update_layout(
            title="Interactive Confusion Matrix (Fashion MNIST)",
            xaxis_title="Predicted Label",
            yaxis_title="True Label"
        )

        # Show interactive plot
        fig.show()

        # Save to HTML and log to wandb
        html_file = "interactive_confusion_matrix.html"
        fig.write_html(html_file)
        with open(html_file, "r") as f:
            html_content = f.read()
            wandb.log({"interactive_confusion_matrix": wandb.Html(html_content)})
    except Exception as test_error:
        print("Error during test evaluation:", test_error)

#############################
# Hyperparameter Analysis (Question 6)
#############################
def analyze_hyperparameters(project, entity):
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        if not runs:
            print("No runs found for analysis.")
            return
        records = []
        for run in runs:
            try:
                config = run.config
                summary = run.summary._json_dict
                record = {
                    "epochs": config.get("epochs", None),
                    "batch_size": config.get("batch_size", None),
                    "num_layers": config.get("num_layers", None),
                    "hidden_size": config.get("hidden_size", None),
                    "learning_rate": config.get("learning_rate", None),
                    "optimizer": config.get("optimizer", None),
                    "weight_decay": config.get("weight_decay", None),
                    "val_loss": summary.get("final_val_loss", None),
                    "val_accuracy": summary.get("final_val_accuracy", None)
                }
                records.append(record)
            except Exception as e:
                print("Error processing a run:", e)
                continue
        if not records:
            print("No valid run records found.")
            return
        df = pd.DataFrame(records)
        fig_parallel = px.parallel_coordinates(
            df,
            dimensions=["epochs", "batch_size", "num_layers", "hidden_size", "learning_rate", "weight_decay", "val_loss", "val_accuracy"],
            color="val_loss",
            color_continuous_scale=px.colors.diverging.Tealrose,
            title="Parallel Coordinates Plot for Hyperparameter Analysis"
        )
        fig_parallel.write_html("parallel_coordinates.html")
        wandb.log({"parallel_coordinates": wandb.Html(open("parallel_coordinates.html", "r").read())})
        fig_parallel.show()
        corr = df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap of Hyperparameters and Metrics")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        wandb.log({"correlation_heatmap": wandb.Image("correlation_heatmap.png")})
        best_runs = df.sort_values("val_loss").head(5)
        print("Top 5 configurations based on validation loss:")
        print(best_runs)
    except Exception as analysis_error:
        print("Error during hyperparameter analysis:", analysis_error)

#############################
# Sweep & Additional Functions
#############################
def sweep_run():
    os.environ["SWEEP_RUN"] = "true"
    main()

def compare_losses(args):
    x_train, y_train, x_val, y_val, _, _ = load_data_80_10_10(args.dataset)
    config = vars(args)
    config["epochs"] = 5
    layers = [784] + [args.hidden_size] * args.num_layers + [10]
    model_ce = NeuralNetwork(layer_sizes=layers,
                             activation=args.activation,
                             weight_init=args.weight_init,
                             weight_decay=args.weight_decay,
                             output_activation="softmax")
    optimizer_ce = Optimizer(optimizer=args.optimizer,
                             lr=args.learning_rate,
                             momentum=args.momentum,
                             beta=args.beta,
                             beta1=args.beta1,
                             beta2=args.beta2,
                             epsilon=args.epsilon,
                             weight_decay=args.weight_decay,
                             clip_value=5.0)
    trainer_ce = Trainer(model_ce, optimizer_ce, config, "cross_entropy", x_train, y_train, x_val, y_val)
    print("Training model with cross-entropy loss...")
    trainer_ce.train()
    activations_ce = model_ce.forward_pass(x_val)
    loss_ce = cross_entropy_loss(y_val, activations_ce[-1])
    acc_ce = accuracy(y_val, activations_ce[-1])

    model_mse = NeuralNetwork(layer_sizes=layers,
                              activation=args.activation,
                              weight_init=args.weight_init,
                              weight_decay=args.weight_decay,
                              output_activation="identity")
    optimizer_mse = Optimizer(optimizer=args.optimizer,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              beta=args.beta,
                              beta1=args.beta1,
                              beta2=args.beta2,
                              epsilon=args.epsilon,
                              weight_decay=args.weight_decay,
                              clip_value=5.0)
    trainer_mse = Trainer(model_mse, optimizer_mse, config, "mean_squared_error", x_train, y_train, x_val, y_val)
    print("Training model with mean squared error loss...")
    trainer_mse.train()
    activations_mse = model_mse.forward_pass(x_val)
    loss_mse = mean_squared_error(y_val, activations_mse[-1])
    acc_mse = accuracy(y_val, activations_mse[-1])
    print(f"Final Validation Loss:\nCross-Entropy: {loss_ce:.4f}, Accuracy: {acc_ce:.4f}\nMSE: {loss_mse:.4f}, Accuracy: {acc_mse:.4f}")

    # Quick bar plot comparison
    losses = [loss_ce, loss_mse]
    accuracies = [acc_ce, acc_mse]
    labels = ['Cross-Entropy', 'MSE']
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.bar(labels, losses, color=['blue', 'orange'])
    plt.ylabel("Validation Loss")
    plt.title("Loss Comparison")
    plt.subplot(1,2,2)
    plt.bar(labels, accuracies, color=['green', 'red'])
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy Comparison")
    plt.tight_layout()
    #plt.show()
    wandb.log({"loss_comparison": plt})

def run_mnist_recommendations(args):
    # Recommended configurations for MNIST
    recommended_configs = [
        {"num_layers": 5, "hidden_size": 64, "learning_rate": 1e-3, "optimizer": "nadam", "activation": "tanh", "weight_decay": 0, "epochs": 10, "batch_size": 64},
        {"num_layers": 3, "hidden_size": 64, "learning_rate": 1e-3, "optimizer": "rmsprop", "activation": "ReLU", "weight_decay": 0, "epochs": 5, "batch_size": 32},
        {"num_layers": 3, "hidden_size": 64, "learning_rate": 1e-3, "optimizer": "adam", "activation": "ReLU", "weight_decay": 0, "epochs": 10, "batch_size": 32}
    ]
    results = []
    # For each recommended configuration, create a separate wandb run.
    for i, rec in enumerate(recommended_configs):
        print(f"\nRunning MNIST experiment configuration {i+1}")
        # Update arguments with current configuration
        args.dataset = "mnist"
        args.num_layers = rec["num_layers"]
        args.hidden_size = rec["hidden_size"]
        args.learning_rate = rec["learning_rate"]
        args.optimizer = rec["optimizer"]
        args.activation = rec["activation"]
        args.weight_decay = rec["weight_decay"]
        args.epochs = rec["epochs"]
        args.batch_size = rec["batch_size"]
        
        # Map loss to output_activation
        output_activation = "softmax" if args.loss == "cross_entropy" else "identity"
        layers = [784] + [args.hidden_size] * args.num_layers + [10]
        x_train, y_train, x_val, y_val, x_test, y_test = load_data_80_10_10(args.dataset)
        
        # Start a new wandb run for this configuration
        run_prefix = f"mnist_config{i+1}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=run_prefix, reinit=True)
        cfg = wandb.config
        
        config = {
            "wandb_project": cfg.wandb_project,
            "wandb_entity": cfg.wandb_entity,
            "dataset": cfg.dataset,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "loss": cfg.loss,
            "optimizer": cfg.optimizer,
            "learning_rate": cfg.learning_rate,
            "momentum": cfg.momentum,
            "beta": cfg.beta,
            "beta1": cfg.beta1,
            "beta2": cfg.beta2,
            "epsilon": cfg.epsilon,
            "weight_decay": cfg.weight_decay,
            "weight_init": cfg.weight_init,
            "num_layers": cfg.num_layers,
            "hidden_size": cfg.hidden_size,
            "activation": cfg.activation,
            "layers": layers,
            "output_activation": "softmax" if cfg.loss == "cross_entropy" else "identity",
            "run_name": run_prefix
        }
        
        model = NeuralNetwork(
            layer_sizes=layers,
            activation=cfg.activation,
            weight_init=cfg.weight_init,
            weight_decay=cfg.weight_decay,
            output_activation="softmax" if cfg.loss == "cross_entropy" else "identity"
        )
        
        optimizer_obj = Optimizer(
            optimizer=cfg.optimizer,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            beta=cfg.beta,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            epsilon=cfg.epsilon,
            weight_decay=cfg.weight_decay,
            clip_value=5.0
        )
        
        # Pass the test set as well to log test accuracy vs. epoch
        trainer = Trainer(model, optimizer_obj, config, cfg.loss, x_train, y_train, x_val, y_val, x_test, y_test)
        trainer.train()
        activations = model.forward_pass(x_test)
        test_acc = accuracy(y_test, activations[-1])
        print(f"Configuration {i+1} Test Accuracy: {test_acc*100:.2f}%")
        wandb.run.summary["test_accuracy"] = test_acc
        wandb.finish()
        results.append((rec, test_acc))
    return results

#############################
# Main
#############################
def main():
    parser = argparse.ArgumentParser(description="Train a feedforward neural network with wandb logging.")
    parser.add_argument("-wp", "--wandb_project", default="neural-network-sweep", help="Weights & Biases project name")
    parser.add_argument("-we", "--wandb_entity", default="cs24m044-iit-madras-alumni-association", help="Weights & Biases entity name")
    parser.add_argument("-d", "--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("-o", "--optimizer", default="adam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum for momentum-based optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for RMSprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for Adam/Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for Adam/Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005, help="Weight decay (L2 regularization)")
    parser.add_argument("-w_i", "--weight_init", default="xavier", choices=["random", "xavier"], help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of neurons per hidden layer")
    parser.add_argument("-a", "--activation", default="ReLU", choices=["identity", "sigmoid", "tanh", "ReLU"], help="Activation function for hidden layers")
    parser.add_argument("--compare_losses", action="store_true", help="Run detailed experiment comparing loss functions (Q8)")
    parser.add_argument("--mnist_recommend", action="store_true", help="Run MNIST recommendation experiments (Q10)")
    parser.add_argument("--sweep", action="store_true", help="Run wandb sweep experiment (Q4)")
    parser.add_argument("--analyze", action="store_true", help="Run hyperparameter analysis (parallel coordinates & correlation) (Q6)")
    
    args, unknown = parser.parse_known_args()
    
    run_prefix = f"hl_{args.num_layers}_bs_{args.batch_size}_ac_{args.activation.lower()}"
    
    if os.environ.get("SWEEP_RUN") == "true":
        args.sweep = False
    
    if args.analyze:
        analyze_hyperparameters(args.wandb_project, args.wandb_entity)
        return
    
    if args.sweep:
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "final_val_loss", "goal": "minimize"},
            "parameters": {
                "epochs": {"values": [5, 10]},
                "num_layers": {"values": [3, 4, 5]},
                "hidden_size": {"values": [32, 64, 128]},
                "weight_decay": {"values": [0, 0.0005, 0.5]},
                "learning_rate": {"values": [1e-3, 1e-4]},
                "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
                "batch_size": {"values": [16, 32, 64]},
                "weight_init": {"values": ["random", "xavier"]},
                "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
                "loss": {"values": ["mean_squared_error"]}
            }
        }
        sweep_config["program"] = "train.py"
        sweep_id_file = "sweep_id.txt"
        if os.path.exists(sweep_id_file):
            with open(sweep_id_file, "r") as f:
                sweep_id = f.read().strip()
            print(f"Using existing sweep id: {sweep_id}")
        else:
            sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
            with open(sweep_id_file, "w") as f:
                f.write(sweep_id)
            print(f"Sweep created with id: {sweep_id}")
        print(f"Running agent for sweep id: {sweep_id}")
        wandb.agent(sweep_id, entity=args.wandb_entity, project=args.wandb_project, function=sweep_run)
        return
    
    if args.compare_losses:
        config = vars(args)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config, resume="never")
        compare_losses(args)
        wandb.finish()
        return
    
    if args.mnist_recommend:
        # Run 3 separate wandb runs for the 3 configurations.
        results = run_mnist_recommendations(args)
        print("\nMNIST Recommendation Experiment Results:")
        for rec, acc in results:
            print(f"Config: {rec} -> Test Accuracy: {acc*100:.2f}%")
        return
    
    # Regular training run
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), resume="never")
    cfg = wandb.config
    run_prefix = f"hl_{cfg.num_layers}_bs_{cfg.batch_size}_ac_{cfg.activation.lower()}"
    wandb.run.name = run_prefix
    wandb.run.save()
    
    # Build layers and load data
    layers = [784] + [cfg.hidden_size] * cfg.num_layers + [10]
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_80_10_10(cfg.dataset)
    
    config = {
        "wandb_project": cfg.wandb_project,
        "wandb_entity": cfg.wandb_entity,
        "dataset": cfg.dataset,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "loss": cfg.loss,
        "optimizer": cfg.optimizer,
        "learning_rate": cfg.learning_rate,
        "momentum": cfg.momentum,
        "beta": cfg.beta,
        "beta1": cfg.beta1,
        "beta2": cfg.beta2,
        "epsilon": cfg.epsilon,
        "weight_decay": cfg.weight_decay,
        "weight_init": cfg.weight_init,
        "num_layers": cfg.num_layers,
        "hidden_size": cfg.hidden_size,
        "activation": cfg.activation,
        "layers": layers,
        "output_activation": "softmax" if cfg.loss == "cross_entropy" else "identity",
        "run_name": run_prefix
    }
    
    model = NeuralNetwork(
        layer_sizes=layers,
        activation=cfg.activation,
        weight_init=cfg.weight_init,
        weight_decay=cfg.weight_decay,
        output_activation="softmax" if cfg.loss == "cross_entropy" else "identity"
    )
    
    optimizer_obj = Optimizer(
        optimizer=cfg.optimizer,
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        beta=cfg.beta,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        epsilon=cfg.epsilon,
        weight_decay=cfg.weight_decay,
        clip_value=5.0
    )
    
    trainer = Trainer(model, optimizer_obj, config, cfg.loss, x_train, y_train, x_val, y_val, x_test, y_test)
    trainer.train()
    test_and_plot(model, cfg.loss, x_test, y_test)
    
    wandb.finish()
    time.sleep(1)

if __name__ == "__main__":
    main()
