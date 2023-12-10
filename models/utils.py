# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
)

def load_data(path, date, lf=10, pr=1):
    labels_pre = np.genfromtxt(
        path + "labels/" + date + "_graph_label_" + str(pr) + ".txt"
    )

    features = np.load(
        path + "features_" + str(lf) + "/" + date + "_features_" + str(lf) + ".npy"
    )

    edges_0 = np.genfromtxt(path + "edges/" + date + "_graph.txt", dtype=np.int32)
    
    if len(edges_0.shape) == 1:
        edges_0 = np.expand_dims(edges_0, axis=0)
    
    edges_1 = np.concatenate(([edges_0[:, 1]], [edges_0[:, 0]]), axis=0).T

    edges = np.concatenate((edges_0, edges_1), axis=0)
    edges = torch.LongTensor(edges)

    idx_train = torch.tensor(np.arange(edges.shape[0]), dtype=torch.long)
    idx_val = idx_train  # .copy()
    idx_test = idx_train  # .copy()

    features = torch.FloatTensor(np.array(features))
    labels_pre = torch.LongTensor(labels_pre + 1)
    return edges, features, labels_pre, idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def metrics(preds, labels):
    labels = np.array(labels)
    preds = np.array(preds)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return acc, macro_f1

def mae_loss(preds, labels):
    labels = np.array(labels)
    preds = np.array(preds)
    mae = mean_absolute_error(labels, preds)
    return mae
