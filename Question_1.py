import wandb
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

class FashionMNISTLoader:
    def __init__(self):
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        return self.x_train, self.y_train, self.x_test, self.y_test

class FashionMNISTVisualizer:

    def __init__(self, class_names):
        self.class_names = class_names

    def plot_samples(self, x_train, y_train):
        plt.figure(figsize=(10, 10))

        for class_index in range(len(self.class_names)):
            sample_index = np.where(y_train == class_index)[0][0]
            image = x_train[sample_index]
            label = self.class_names[y_train[sample_index]]

            plt.subplot(2, 5, class_index + 1)
            plt.imshow(image, cmap=plt.cm.binary)
            plt.title(label)
            plt.axis('off')

        return plt

class WandbLogger:

    def __init__(self, project, run_name):

        wandb.login()
        self.run = wandb.init(project=project, name=run_name)

    def log_image(self, plt_obj, key="Question 1"):

        wandb.log({key: wandb.Image(plt_obj)})

    def finish(self):

        wandb.finish()

def main():

    data_loader = FashionMNISTLoader()
    x_train, y_train, x_test, y_test = data_loader.load_data()


    visualizer = FashionMNISTVisualizer(data_loader.class_names)
    plot_obj = visualizer.plot_samples(x_train, y_train)


    logger = WandbLogger(
        project="SayanDas_CS24M044_DA6401_DL_Assignment1",
        run_name="Question 1"
    )


    logger.log_image(plot_obj)
    plt.show()


    logger.finish()

if __name__ == '__main__':
    main()
