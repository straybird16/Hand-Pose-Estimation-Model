import glob

import numpy
import numpy as np
import pickle

import torch.cuda
from scipy import ndimage as ndimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy.lib.npyio import NpzFile
import cv2 as cv
import open3d as o3d
import os
import sys
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
# import pypotree
import re
from utils import load_data
from model import *
import random as random

# export PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb:256
# CUDA_LAUNCH_BLOCKING = 1

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')
print(f"Using device: {device}")

# You may need this line if you have more than 1 GPU
torch.cuda.set_device(device)

use_2000_points = False
package_path = './dhg_data.2.pckl' if use_2000_points else './dhg_data.pckl'

# train or test
train = True
# number of global features extracted by the PointNet.
num_global_features = 64
# ---------------------------------------------------------
# Step 6. Train Model
# ---------------------------------------------------------
if train:
    save_best = True
    # import dataset
    train_set, x_test, test_set, y_test = load_data(package_path)
    train_set = train_set.to(device).float()
    test_set = test_set.to(device)

    x_test = x_test.float()
    y_test = y_test - 1

    x_val = x_test[0:100, :]
    y_val = y_test[0:100]
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    print(train_set.shape, test_set.shape, x_val.shape, y_val.shape)

    # --------------------     testing      --------------------

    # -------------------- hyper-parameters --------------------
    num_training = 4
    num_epoch = 80
    batch_size = 32
    learning_rate = 1e-2
    decay = 2e-3
    # --------------------     data-set     --------------------
    # the dataset use 1 indexing, minus 1 from it
    test_set = test_set - 1
    train_set_length = len(train_set)
    # -------------------- initialize model --------------------
    model = PPN(3, num_global_features, hidden_size=num_global_features, device=device).to(device)
    # COMMENT THIS OUT WHEN USING NEW MODEL SETTINGS
    # model.load_state_dict(torch.load('saved_model'))
    # model.load_state_dict(torch.load('./model/saved_model_PN_LSTM_64x3'))

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    print(f"Number of parameters in model: {num_of_parameters}")

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay, nesterov=True, momentum=0.9)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # --------------------     training     --------------------
    model.train(True)

    # if you want to partially train the model, use this
    fine_tune = False
    if fine_tune:
        for param in model.png.point_net.parameters():
            param.requires_grad = True

        for param in model.lstm.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    # lists to generate graphs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    epochs = np.arange(num_training * num_epoch) + 1

    min_loss = 100
    def train_once(min_loss=min_loss):
        for epoch in range(num_epoch):
            torch.cuda.empty_cache()
            print(f"\nEpoch{epoch + 1}:\n")
            # epoch parameters
            # randomize batches
            order = random.sample(list(range(train_set_length)), train_set_length)
            epoch_train_set = train_set[order]
            epoch_test_set = test_set[order]
            # print(f"train set: {train_set.shape}")
            # print(f"epoch_train_set.shape = {epoch_train_set.shape}")

            batches = []
            test_batches = []
            i = 0
            while i < train_set_length:
                next_i = min(i + batch_size, train_set_length)
                batches.append(epoch_train_set[i: next_i])
                test_batches.append(epoch_test_set[i: next_i])
                i = next_i
            num_batch = len(batches)
            # print(f"batch num: {num_batch}\n")

            train_loss = 0
            train_accuracy = 0

            for index, batch in enumerate(batches):
                print(f"processing batch {index + 1}:")
                batch_loss = 0
                optimizer.zero_grad()
                output = model(batch)  # B x 14
                loss = criterion(output, test_batches[index])
                print(f"batch loss = {loss}")
                c = torch.argmax(output, dim=-1)
                print(c)
                accuracy = sum(c == test_batches[index]) / len(test_batches[index])
                print(f"Accuracy = {accuracy}\n")
                train_accuracy += accuracy
                loss.backward()
                train_loss += loss
                optimizer.step()
            print(f"epoch average train loss = {train_loss / num_batch}")
            train_losses.append((train_loss / num_batch).to(cpu).detach().numpy())

            print(f"epoch average train accuracy = {train_accuracy / num_batch}\n")
            train_accuracies.append((train_accuracy / num_batch).to(cpu).detach().numpy())

            with torch.no_grad():
                val_output = model(x_val)
                validation_loss = criterion(val_output, y_val)
                print(f"validation loss = {validation_loss.item()}")
                val_losses.append(validation_loss.to(cpu).detach().numpy())

                val_accuracy = sum(torch.argmax(val_output, dim=-1) == y_val) / 100
                print(f"validation accuracy = {val_accuracy}")
                val_accuracies.append(val_accuracy.to(cpu).detach().numpy())

                if save_best and validation_loss < min_loss:
                    min_loss = validation_loss
                    torch.save(model.state_dict(), 'saved_model')
                    print(f"current min loss = {min_loss}")

        if not save_best:
            torch.save(model.state_dict(), 'saved_model')
        return min_loss


    # train
    cnt = num_training
    min_loss = 10
    while cnt > 0:
        model.partial = True if cnt >= num_training - 1 else False
        # num_epoch = 100 if cnt == num_training and not save_best else 50
        min_loss = train_once(min_loss)
        cnt -= 1
    # py plot
    plt.plot(epochs, train_losses, label="average train loss", linestyle="-.")
    plt.plot(epochs, train_accuracies, label="average train accuracy")
    plt.plot(epochs, val_losses, label="validation loss", linestyle="-.")
    plt.plot(epochs, val_accuracies, label="validation accuracy")
    plt.legend()
    plt.show()

# ---------------------------------------------------------
# Step 7. Test Model
# ---------------------------------------------------------
else:
    print("Testing...................................")
    device = torch.device('cpu')
    _, x_test, _, y_test = load_data(package_path)
    x_test = x_test.float()
    y_test = y_test - 1

    x_test = x_test[100:, :]
    y_test = y_test[100:]
    print(x_test.shape)
    print(y_test.shape)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    model = PPN(3, num_global_features, hidden_size=num_global_features, device=device).to(device)
    model.load_state_dict(torch.load('saved_model'))
    # model.load_state_dict(torch.load('./model/saved_model_PN_LSTM_64x3'))
    criterion = nn.CrossEntropyLoss(reduction='mean')

    model.eval()
    model.partial = False
    loss = 0
    test_accuracy = 0
    # for i, test in enumerate(x_test):
    #   test_output = model(torch.unsqueeze(test, 0))
    #  test_loss = criterion(test_output, torch.unsqueeze(y_test[i], 0))
    # loss += test_loss
    # test_accuracy += sum(torch.argmax(test_output, dim=-1) == y_test[i])

    test_output = model(x_test)
    test_loss = criterion(test_output, y_test)
    print(f"Test loss = {test_loss}")
    pred_y = torch.argmax(test_output, dim=-1)
    test_accuracy = sum(pred_y == y_test) / y_test.size(0)
    print(f"Test accuracy = {test_accuracy}")

    pred_y = pred_y.detach().numpy()
    y_test = y_test.detach().numpy()
    correct_indices = np.argwhere((pred_y == y_test)==False).reshape(-1)
    correct_labels = pred_y[correct_indices]
    print(f"corrected labels: {correct_labels}")
    plt.hist(correct_labels, bins=14, histtype='bar', align='mid')
    plt.xticks(correct_labels)
    plt.xlabel("type of gesture")
    plt.ylabel("number of wrong predictions")
    plt.show()


