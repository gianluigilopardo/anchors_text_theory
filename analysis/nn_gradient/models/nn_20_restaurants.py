"""
Training a 20 layers neural network with Restaurants dataset
"""

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
from torch import optim
import os

from dataset.dataset import Dataset

np.random.seed(42)
torch.manual_seed(42)

# DATA
dir = os.path.join('analysis', 'nn_gradient', 'models')
datapath = os.getcwd().replace(dir, 'dataset')
DATASET = 'restaurants'
data = Dataset(DATASET, datapath)
df, X, y = data.df, data.X, data.y

X_train, X_test, y_train, y_test = train_test_split(X, y)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(norm=None, max_features=1000)
vect = 'tf_idf'

train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# convert to pytorch tensor
x_train = torch.tensor(scipy.sparse.csr_matrix.todense(train_vectors)).float()
x_test = torch.tensor(scipy.sparse.csr_matrix.todense(test_vectors)).float()
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# CLASSIFIER
model = nn.Sequential(nn.Linear(x_train.shape[1], 1000),
                      nn.ReLU(),

                      nn.Linear(1000, 950),
                      nn.ReLU(),

                      nn.Linear(950, 900),
                      nn.ReLU(),

                      nn.Linear(900, 850),
                      nn.ReLU(),

                      nn.Linear(850, 800),
                      nn.ReLU(),

                      nn.Linear(800, 750),
                      nn.ReLU(),

                      nn.Linear(750, 700),
                      nn.ReLU(),

                      nn.Linear(700, 650),
                      nn.ReLU(),

                      nn.Linear(650, 600),
                      nn.ReLU(),

                      nn.Linear(600, 550),
                      nn.ReLU(),

                      nn.Linear(550, 500),
                      nn.ReLU(),

                      nn.Linear(500, 450),
                      nn.ReLU(),

                      nn.Linear(450, 400),
                      nn.ReLU(),

                      nn.Linear(400, 350),
                      nn.ReLU(),

                      nn.Linear(350, 300),
                      nn.ReLU(),

                      nn.Linear(300, 250),
                      nn.ReLU(),

                      nn.Linear(250, 200),
                      nn.ReLU(),

                      nn.Linear(200, 150),
                      nn.ReLU(),

                      nn.Linear(150, 100),
                      nn.ReLU(),

                      nn.Linear(100, 50),
                      nn.ReLU(),

                      nn.Linear(50, 2),
                      nn.LogSoftmax(dim=1))

# Define the loss
criterion = nn.NLLLoss()

# Forward pass, get our logits
logps = model(x_train)
# Calculate the loss with the logits and the labels
loss = criterion(logps, y_train)

loss.backward()

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-10)  # 0.816

# train
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

epochs = 200
best_accuracy = 0
best_model = []
PATH = os.path.join(os.getcwd(), f'nn_20_{DATASET}.p')
for e in range(epochs):
    optimizer.zero_grad()
    output = model.forward(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    train_loss = loss.item()
    train_losses.append(train_loss)
    optimizer.step()

    ps = torch.exp(model(x_train))
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == y_train.view(*top_class.shape)
    train_accuracy = torch.mean(equals.float())
    train_accuracies.append(train_accuracy)

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        log_ps = model(x_test)
        test_loss = criterion(log_ps, y_test)
        test_losses.append(test_loss)

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y_test.view(*top_class.shape)
        test_accuracy = torch.mean(equals.float())
        test_accuracies.append(test_accuracy)

    model.train()

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = model
        if best_accuracy >= 0.8:
            torch.save(model.state_dict(), PATH)

    print(f"Epoch: {e + 1}/{epochs}.. ",
          f"Training Loss: {train_loss:.3f}.. ",
          f"Training Accuracy: {train_accuracy:.3f}.. ",
          f"Test Loss: {test_loss:.3f}.. ",
          f"Test Accuracy: {test_accuracy:.3f}.. ",
          f"Best Accuracy: {best_accuracy:.3f}")



