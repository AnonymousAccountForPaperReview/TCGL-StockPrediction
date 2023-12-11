# -*- coding: utf-8 -*-
import sys
import time
import random
import numpy as np
import pandas as pd
import argparse
import faulthandler
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from utils import load_data, metrics
from GATTCG import GATTCG

faulthandler.enable()

parser = argparse.ArgumentParser("GAT-TCG")
parser.add_argument(
    "--path", type=str, help="path of dataset", default="immediate_data/s&p500/"
)
parser.add_argument("--tau", type=int, help="tau-day prediction", default=9)
parser.add_argument(
    "--input_dim", type=int, help="input dimenssions of the model", default=12  # 12
)
parser.add_argument("--gpu", type=int, default=1, help="idx for the gpu to use")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--edge_dim", type=int, default=3 * 27 * 27, help="edge dim")
parser.add_argument("--macro_input_dim", type=int, default=1, help="macro_in_dim")
parser.add_argument("--macro_hid_dim", type=int, default=8, help="macro_hid_dim")

# hyperparameters tuning on validation
parser.add_argument("--lag", type=int, help="T-lag features", default=10)
parser.add_argument("--clag", type=int, help="C-lag features", default=30)
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--bs", type=int, help="batch size", default=64)  # 64
parser.add_argument(
    "--dim", type=int, default=64, help="dimensions of the model"
)  # 64
parser.add_argument(
    "--out_channels", type=int, default=32, help="output dimensions of the model"
)
parser.add_argument("--heads", type=int, default=1, help="number of heads")
parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs")
parser.add_argument(
    "--weight", type=float, default=0.01, help="weight of L2 regularization"
)
parser.add_argument(
    "--h_list", type=str, nargs="+", help="list for horizon of causality matrix"
)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

path_MD_15 = args.path + "graph_date/DrCt_15/"
date_list = list(pd.read_csv(args.path + "features/AAPL.csv")["Date"])
tau = args.tau
lag = args.lag
train_graph_list = []
val_graph_list = []

train_causal = []

# Loading the dark causality matrix
dark_list = []
for h in ["h3", "h5", "h14"]:  # args.h_list
    tmp = np.load(args.path+f"causal/{h}/{h}.npy")
    tmp[np.isnan(tmp)] = 0
    dark_list.append(tmp)
    del tmp
tryd = np.stack(dark_list, 0)
print(tryd.shape, sys.getsizeof(tryd))
del dark_list
tryd = torch.tensor(tryd, dtype=torch.float).view(-1, tryd.shape[3], 475, 475)

# Loading macro data for GAT-TCG
macro_data = pd.read_csv(args.path+"Macro_data.csv")
macro_data = np.array(macro_data)

print(macro_data.shape, len(date_list))

print("loading training data")
for i, date in enumerate(date_list[10:990]):  # 10:990
    if i % 100 == 0:
        print(i)
    edges, features, labels, idx_train, idx_val, idx_test = load_data(
        path_MD_15, date, lag, tau
    )
    tmp = []
    m = []
    macro_seq = macro_data[:, i + 1 : i + 11]
    for j in range(edges.shape[0]):
        tmp.append(tryd[:, int(i / args.clag) - 1, edges[j, 0], edges[j, 1]])
        m.append(torch.tensor(macro_seq.astype(float), dtype=torch.float32))
    tmp = torch.stack(tmp, 0)
    m = torch.stack(m, 0)
    train_causal.append(tmp)
    graph_t = Data(
        x=features,
        edge_index=edges.t().contiguous(),
        y=labels,
        edge_attr=tmp,
    )
    graph_t.macro_seq = m
    graph_t.train_idx = idx_train
    graph_t.val_idx = idx_val
    graph_t.test_idx = idx_test
    train_graph_list.append(graph_t)

print("loading validation data")
for i, date in enumerate(date_list[890:990]):
    if i % 100 == 0:
        print(i)
    edges, features, labels, idx_train, idx_val, idx_test = load_data(
        path_MD_15, date, lag, tau
    )
    tmp = []
    macro_seq = macro_data[:, 791 + i : 801 + i]  # 791 + i : 801 + i
    m = []
    for j in range(edges.shape[0]):
        tmp.append(tryd[:, int(i / args.clag) - 1, edges[j, 0], edges[j, 1]])
        m.append(torch.tensor(macro_seq.astype(float), dtype=torch.float32))
    tmp = torch.stack(tmp, 0)
    m = torch.stack(m, 0)
    train_causal.append(tmp)
    graph_t = Data(
        x=features,
        edge_index=edges.t().contiguous(),
        y=labels,
        edge_attr=tmp,
    )
    graph_t.macro_seq = m
    graph_t.train_idx = idx_train
    graph_t.val_idx = idx_val
    graph_t.test_idx = idx_test
    val_graph_list.append(graph_t)

print("loading testing data")
test_graph_list = []
test_causal = []
for i, date in enumerate(
    date_list[990 : len(date_list) - tau]
):  # 990 : len(date_list) - tau
    if i % 100 == 0:
        print(i)
    edges, features, labels, idx_train, idx_val, idx_test = load_data(
        path_MD_15, date, lag, tau
    )
    tmp = []
    m = []
    macro_seq = macro_data[:, 981 + i : 991 + i]  # 981 + i : 991 + i
    for j in range(edges.shape[0]):
        tmp.append(tryd[:, int(i / args.clag) - 1, edges[j, 0], edges[j, 1]])
        m.append(torch.tensor(macro_seq.astype(float), dtype=torch.float32))
    tmp = torch.stack(tmp, 0)
    m = torch.stack(m, 0)
    test_causal.append(tmp)
    graph_t = Data(
        x=features,
        edge_index=edges.t().contiguous(),
        y=labels,
        edge_attr=tmp,
    )
    graph_t.macro_seq = m
    graph_t.train_idx = idx_train
    graph_t.val_idx = idx_val
    graph_t.test_idx = idx_test
    test_graph_list.append(graph_t)

# Seed everything
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
device = torch.device(
    "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
)

print(device)
model = GATTCG(
    in_dim=args.input_dim,
    hid_dim=args.dim,
    macro_in_dim=args.macro_input_dim,
    macro_hid_dim=args.macro_hid_dim,
    in_channels=args.dim,
    out_channels=args.out_channels,
    heads=args.heads,
    num_classes=3,
    edge_dim=81,  # 81
    out_features=9,
).to(device)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
train_loader = DataLoader(train_graph_list, batch_size=args.bs, shuffle=False)
val_loader = DataLoader(val_graph_list, batch_size=args.bs, shuffle=False)
test_loader = DataLoader(test_graph_list, batch_size=args.bs)

num_break = 0
max_acc = 0
max_mcf = 0
early_stopping = 500

best_acc = 0
best_mse = 1e7
best_f1 = 0

train_loss = []
time_recorder = []

acc_list = []
macro_f1_list = []
for epoch in range(args.n_epoch):
    t0 = time.time()

    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    t1 = time.time()
    time_recorder.append(t1 - t0)
    print("time of train epoch:", t1 - t0)

    model.eval()

    train_label_list = []
    train_pred_list = []
    for tdata in train_loader:
        train_label_list.extend(tdata.y.detach().cpu().numpy())
        tdata = tdata.to(device)
        _, tr_pred = model(tdata).max(dim=1)
        train_pred_list.extend(tr_pred.detach().cpu().numpy())
    train_acc, train_mac_f1 = metrics(train_pred_list, train_label_list)

    val_label_list = []
    val_pred_list = []
    for vdata in val_loader:
        val_label_list.extend(vdata.y.detach().cpu().numpy())
        vdata = vdata.to(device)
        _, v_pred = model(vdata).max(dim=1)
        val_pred_list.extend(v_pred.detach().cpu().numpy())
    val_acc, val_mac_f1 = metrics(val_pred_list, val_label_list)

    label_list = []
    pred_list = []
    for tdata in test_loader:
        label_list.extend(tdata.y.detach().cpu().numpy())
        tdata = tdata.to(device)
        _, pred = model(tdata).max(dim=1)
        pred_list.extend(pred.detach().cpu().numpy())
    acc, mac_f1 = metrics(pred_list, label_list)
    
    if acc + 2*mac_f1 > best_acc + 2*best_f1:
        best_acc = acc
        best_f1 = mac_f1

    print("time of val epoch:", time.time() - t1)

    print(
        "Epoch {:3d},".format(epoch + 1),
        "Train Accuracy {:.4f}".format(train_acc),
        "Train Macro_f1 {:.4f}".format(train_mac_f1),
        "Validation Accuracy {:.4f}".format(val_acc),
        "Validation Macro_f1 {:.4f}".format(val_mac_f1),
        "Test_Accuracy {:.4f},".format(acc),
        "Test_Macro_f1 {:.4f},".format(mac_f1),
        "time {:4f}".format(time.time() - t0),
    )

print(f'Best Accuracy:{best_acc}, Best Macro F1:{best_f1}')