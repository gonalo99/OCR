import copy
import glob
import cv2
import numpy as np
import random
import pickle
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import utils
import time


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability
        As in logistic regression, the __init__ here defines a bunch of
        attributes that each FeedforwardNetwork instance has. Note that nn
        includes modules for several activation functions and dropout as well.
        """
        super().__init__()
        # Implement me!
        if activation_type == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        list = []
        for i in range(layers):
            if i == 0:
                list.append(nn.Linear(n_features, hidden_size))
            else:
                list.append(nn.Linear(hidden_size, hidden_size))
            list.append(activation)
            #list.append(nn.Dropout(0.3))
        list.append(nn.Linear(hidden_size, n_classes))

        self.model = nn.Sequential(*list)

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        This method needs to perform all the computation needed to compute
        the output logits from x. This will include using various hidden
        layers, pointwise nonlinear functions, and dropout.
        """
        return self.model(x.float())


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.
    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.
    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """

    optimizer.zero_grad()
    yhat = model(X.float())
    loss = criterion(yhat, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X.float())  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, train_data):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    X, y = processData(train_data)
    y_hat = predict(model, torch.tensor(X))
    n_correct = (torch.tensor(y) == y_hat).sum().item()
    n_possible = len(X)
    model.train()
    return n_correct / n_possible

def main():
    train_data = buildTrainingData()
    #with open("train_data", "wb") as fp:  # Pickling
    #    pickle.dump(train_data, fp)
    return

    with open("train_data", "rb") as fp:  # Unpickling
        train_data = pickle.load(fp)

    configure_seed(seed=42)
    n_classes = 2
    n_feats = 9

    hidden_sizes = 20
    layers = 5
    activation = 'relu'
    optimizer = 'adam'
    learning_rate = 0.00005

    l2_decay = 0
    epochs = 2000

    model = FeedforwardNetwork(
            n_classes, n_feats,
            hidden_sizes, layers,
            activation)

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=learning_rate,
        weight_decay=l2_decay)

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, epochs + 1)
    train_mean_losses = []
    train_losses = []

    best_acc = 0
    for ii in epochs:
        random.shuffle(train_data)
        x, y = processData(train_data)
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in zip(torch.tensor(x), torch.tensor(y)):
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        acc = evaluate(model, train_data)
        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))
        print('Accuracy:', acc)

        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())

        print("Best:", best_acc)

    print("Best:", best_acc)
    torch.save(best_model_state, 'checkpoint.pth')


def processData(list):
    x = []
    y = []
    for data in list:
        x.append(data[:-1])
        y.append(data[-1])

    return x, y


def buildTrainingData():
    train_data = []

    # Loop through all the gt files
    for filename in sorted(glob.glob('../data/custom_detection/gts/*.txt')):
        # Get the respective frame
        img_no = filename[33:-3]
        frame = cv2.imread('../data/custom_detection/images/img_' + img_no + 'png')

        # Get each of the frame's crops
        for line in open(filename, "r").readlines():
            data = line.split(",")
            box = np.array([[float(data[0]), float(data[1])], [float(data[2]), float(data[3])], [float(data[4]), float(data[5])], [float(data[6]), float(data[7])]])
            quality = data[9]

            new_box = [box[3], box[0], box[1], box[2]]
            cropped = utils.fourPointsTransform(frame, new_box, 1)

            # Add features to training data
            if quality[:-1] == 'good':
                sign = 1
            else:
                sign = 0

            train_data.append([getGradient(frame), getGradient(cropped), getLaplacian(frame), getLaplacian(cropped), getSharpEdges(frame), getSharpEdges(cropped), getAverageGradient(cropped), getAverageLaplacian(cropped), getAverageSharpEdges(cropped), sign])

    scaler = StandardScaler()
    scaler.fit(train_data)
    print(scaler.mean_)
    #train_data = scaler.fit_transform(train_data)
    #print(train_data)

    return train_data


def getGradient(image):
    image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    gX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    return np.average(np.sqrt((gX ** 2) + (gY ** 2)))


def getLaplacian(image):
    image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return cv2.Laplacian(image, cv2.CV_64F).var()


def getAverageGradient(image):
    image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    gX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    height, width = image.shape[:2]

    return np.average(np.sqrt((gX ** 2) + (gY ** 2)))/(width*height)


def getAverageLaplacian(image):
    image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    height, width = image.shape[:2]
    return cv2.Laplacian(image, cv2.CV_64F).var()/(width*height)


def getSharpEdges(image):
    image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    image_canny = cv2.Canny(image, 175, 225)
    nonzero_ratio = np.count_nonzero(image_canny) * 1000.0 / image_canny.size
    return nonzero_ratio


def getAverageSharpEdges(image):
    height, width = image.shape[:2]
    image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    image_canny = cv2.Canny(image, 175, 225)
    nonzero_ratio = np.count_nonzero(image_canny) * 1000.0 / image_canny.size
    return nonzero_ratio/(width*height)


if __name__ == "__main__":
    main()
