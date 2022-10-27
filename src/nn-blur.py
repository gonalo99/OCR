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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

import utils
import time


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
            list.append(nn.Dropout(0.5))
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


def evaluate(model, test_data, scaler, pca):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    X, y = processData(test_data)
    X = scaler.transform(X)
    X = pca.transform(X)
    y_hat = predict(model, torch.tensor(X))
    n_correct = (torch.tensor(y) == y_hat).sum().item()
    n_possible = len(X)
    model.train()
    return n_correct / n_possible


def main():
    #train_data = buildTrainingData()
    #with open("train_data", "wb") as fp:  # Pickling
    #    pickle.dump(train_data, fp)
    #return

    # Retrieve training data
    with open("train_data", "rb") as fp:  # Unpickling
        train_data = pickle.load(fp)

    # Configure NN model
    n_classes = 2
    n_feats = 5
    hidden_sizes = 201
    layers = 2
    activation = 'relu'
    optimizer = 'adam'
    learning_rate = 0.000005
    l2_decay = 0
    epochs = 5000
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

    # Split train and test set
    x, y = processData(train_data)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=109)  # 70% training and 30% test
    test_data = [x+[y] for x,y in zip(X_test, y_test)]

    # Feature scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)

    #fig = plt.figure(figsize=(8, 8))
    #ax = plt.axes(projection='3d')
    #ax.set_xlabel('Principal Component 1', fontsize=15)
    #ax.set_ylabel('Principal Component 2', fontsize=15)
    #ax.set_title('2 component PCA', fontsize=20)
    #targets = ['0', '1']
    #colors = ['r', 'g']
    #for target, color in zip(targets,colors):
    #    indicesToKeep = [i for i,e in enumerate(y_train) if e == int(target)]
    #    ax.scatter(principalComponents[indicesToKeep, 0], principalComponents[indicesToKeep, 1], principalComponents[indicesToKeep, 2], c=color, s=50)
    #ax.legend(targets)
    #ax.grid()
    #plt.show()
    #return

    # training loop
    epochs = torch.arange(1, epochs + 1)
    train_losses = []
    accuracies = []
    mean_losses = []
    best_acc = 0
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in zip(torch.tensor(X_train), torch.tensor(y_train)):
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        acc = evaluate(model, test_data, scaler, pca)
        accuracies.append(acc)
        mean_loss = torch.tensor(train_losses).mean().item()
        mean_losses.append(mean_loss)
        print('Training loss: %.4f' % mean_loss)
        print('Accuracy:', acc)

        if acc > best_acc:
            best_epoch = ii
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())

        print("Best:", best_acc, "in epoch", best_epoch)

    print("Best:", best_acc, "in epoch", best_epoch)
    yhat = savgol_filter(accuracies, 17, 4)
    # Plot training loss and test accuracy in same graph
    fig, ax = plt.subplots()
    ax.plot(epochs[20:], mean_losses[20:], label='Loss', color='blue')
    #ax2 = ax.twinx()
    #ax2.plot(epochs[20:], accuracies[20:], label='Accuracy', color='red')
    ax3 = ax.twinx()
    ax3.plot(epochs[20:], yhat[20:], label='Smoothed Accuracy', color='orange')
    fig.legend()
    plt.show()

    # Saved model has an accuracy of 0.9437 %
    torch.save(best_model_state, 'checkpoint.pth')


def processData(data):
    x = []
    y = []
    for element in data:
        x.append(element[:-1])
        y.append(element[-1])

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
