# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

dataset = 's&p500' # csi300 or s&p500
raw_path = f"datasets/{dataset}/"  
path = f"immediate_data/{dataset}/" # to save results

os.system(
    f"rm -rf {path}/graph_date/DrCt_15"
)
os.system(
    f"mkdir {path}/graph_date/DrCt_15"
)

if not os.path.exists(path + "graph_date/DrCt_15/edges"):
    os.makedirs(path + "graph_date/DrCt_15/edges")

if not os.path.exists(path + "graph_date/DrCt_15/Adjs"):
    os.makedirs(path + "graph_date/DrCt_15/Adjs")

if not os.path.exists(path + "graph_date/DrCt_15/features_10"):
    os.makedirs(path + "graph_date/DrCt_15/features_10")

if not os.path.exists(path + "graph_date/DrCt_15/labels"):
    os.makedirs(path + "graph_date/DrCt_15/labels")

date = pd.read_csv(raw_path + "date.csv")
DrCt_15 = np.load(path + "graph/DrCt_15.npy")
edges = []


def file_name():
    with open(raw_path + "stock_symbols.csv", "r", encoding="utf-8") as fr:
        stock_symbols = fr.readlines()
        stock_symbols = [sy.strip() + ".csv" for sy in stock_symbols]
    return stock_symbols


files = file_name()

symbol_features = {}

s_t = time.time()
for file in files:
    features_df = pd.read_csv(path + "features/" + file)
    symbol_features[file] = features_df

# for s&p500 dataset
# data of first 34 days were used to calculate the indicators
date_15 = pd.read_csv(path + "features/AAPL.csv")["Date"]

print(date_15.shape)
Adj = []

tou = 9 
P_num_edge = []
N_num_edge = []
for i in tqdm(range(tou, 1468 - tou)):
    date_feature_10 = []

    label_aux = []
    num_edge = 0

    adj = np.zeros((475, 475))
    all_data = DrCt_15[i, :, :].reshape(1, -1)

    p = np.percentile(all_data, 96)
    with open(
        path + "graph_date/DrCt_15/edges/" + str(date_15.loc[i]) + "_graph.txt",
        "w",
        encoding="utf-8",
    ) as fw:
        for j in range(475):
            for k in range(0, 475):
                if DrCt_15[i, j, k] >= p:
                    if k != j:
                        adj[j, k] = 1
                        num_edge = num_edge + 1
                    if k >= j + 1:
                        fw.write(str(j) + " " + str(k) + "\n")

    np.save(path + "graph_date/DrCt_15/Adjs/" + str(date_15.loc[i]) + "_Adj.npy", adj)

    with open(
        path
        + "graph_date/DrCt_15/labels/"
        + str(date_15.loc[i])
        + "_graph_label_1.txt",
        "w",
        encoding="utf-8",
    ) as fw_1:
        with open(
            path
            + "graph_date/DrCt_15/labels/"
            + str(date_15.loc[i])
            + "_graph_label_3.txt",
            "w",
            encoding="utf-8",
        ) as fw_3:
            with open(
                path
                + "graph_date/DrCt_15/labels/"
                + str(date_15.loc[i])
                + "_graph_label_5.txt",
                "w",
                encoding="utf-8",
            ) as fw_5:
                with open(
                    path
                    + "graph_date/DrCt_15/labels/"
                    + str(date_15.loc[i])
                    + "_graph_label_7.txt",
                    "w",
                    encoding="utf-8",
                ) as fw_7:
                    with open(
                        path
                        + "graph_date/DrCt_15/labels/"
                        + str(date_15.loc[i])
                        + "_graph_label_9.txt",
                        "w",
                        encoding="utf-8",
                    ) as fw_9:
                        with open(
                            path
                            + "graph_date/DrCt_15/labels/"
                            + str(date_15.loc[i])
                            + "_graph_label_aux.txt",
                            "w",
                            encoding="utf-8",
                        ) as fl:
                            for file in files:
                                features_df = symbol_features[file]
                                label_1 = features_df.loc[i, "label_1"]
                                label_3 = features_df.loc[i, "label_3"]
                                label_5 = features_df.loc[i, "label_5"]
                                label_7 = features_df.loc[i, "label_7"]
                                label_9 = features_df.loc[i, "label_9"]
                                label_aux = features_df.loc[i, "label_auxiliary"]
                                fw_1.write(str(int(label_1)) + "\n")
                                fw_3.write(str(int(label_3)) + "\n")
                                fw_5.write(str(int(label_5)) + "\n")
                                fw_7.write(str(int(label_7)) + "\n")
                                fw_9.write(str(int(label_9)) + "\n")
                                fl.write(str(int(label_aux)) + "\n")
                                date_feature_10.append(
                                    features_df.loc[
                                        i - 9 : i,
                                        [
                                            "Open",
                                            "High",
                                            "Low",
                                            "Close",
                                            "Volume",
                                            "MACD",
                                            "RSI",
                                            "SOK",
                                            "WILLR",
                                            "OBV",
                                            "ROC",
                                            "label_auxiliary",
                                        ],
                                    ].values
                                )

    date_feature_10 = np.array(date_feature_10)
    np.save(
        path
        + "graph_date/DrCt_15/features_10/"
        + str(date_15.loc[i])
        + "_features_10.npy",
        date_feature_10,
    )

print("Time elapsed:", time.time() - s_t)
