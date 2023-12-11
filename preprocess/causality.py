# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

dataset = 's&p500' # csi300 or s&p500
raw_path = f"datasets/{dataset}/"  
path = f"immediate_data/{dataset}/" # to save results
os.makedirs(path + "graph/", exist_ok=1)


def file_name(file_dir):
    with open(raw_path + "stock_symbols.csv", "w", encoding="utf-8") as fw:
        for _, _, files in os.walk(file_dir):
            fw.writelines(file.replace(".csv", "") + "\n" for file in files)
    return files


trade_data = pd.read_csv(raw_path + "date.csv")
trade_data.set_index("Date", inplace=True)
files = file_name(path + "pre_process")
date_close = pd.DataFrame()
date_label_1 = pd.DataFrame()
date_label_3 = pd.DataFrame()
date_label_5 = pd.DataFrame()
date_label_7 = pd.DataFrame()
date_label_9 = pd.DataFrame()
for file in files:
    #    print(file)
    pre_stock = pd.read_csv(path + "pre_process/" + file)
    date_close[file.replace(".csv", "")] = np.log(
        pre_stock["Close"] / pre_stock["Close"].shift(1)
    )
    date_label_1[file.replace(".csv", "")] = pre_stock["label_1"]
    date_label_3[file.replace(".csv", "")] = pre_stock["label_3"]
    date_label_5[file.replace(".csv", "")] = pre_stock["label_5"]
    date_label_7[file.replace(".csv", "")] = pre_stock["label_7"]
    date_label_9[file.replace(".csv", "")] = pre_stock["label_9"]


# Darkcausual distance
print("calculating Dark-Causuality distance")
g = np.load(
    path + "causal/h0/h0.npy"
)

st = time.time()
DrCt_reg_15 = []
pos_mul = np.zeros((9, 9))
neg_mul = np.zeros((9, 9))
dark_mul = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        if i == j:
            pos_mul[i, j] = 1
        if i == 9 - j:
            neg_mul[i, j] = 1
        if not (i == j or i == 9 - j):
            dark_mul[i, j] = 1

now = 0
las = 0
drct = np.zeros((475, 475))
for i in tqdm(range(15, len(date_label_1))):
    now = int(i / 30) - 1
    if now == las:
        DrCt_reg_15.append(drct)
        continue
    drct = np.zeros((475, 475))

    for j in range(475):
        for k in range(j + 1, 475):
            pos = np.sum(np.multiply(g[:, :, now, j, k], pos_mul))
            neg = np.sum(np.multiply(g[:, :, now, j, k], neg_mul))
            dark = np.sum(np.multiply(g[:, :, now, j, k], dark_mul))
            tmp = max(pos, neg, dark)
            drct[j, k] = tmp
            drct[k, j] = tmp

    las = now
    DrCt_reg_15.append(drct)
print(time.time() - st)
DrCt_reg_15 = np.array(DrCt_reg_15)
print(DrCt_reg_15.shape)

# print(DrCt_reg_15[18:].shape)
np.save(path + "graph/DrCt_15.npy", DrCt_reg_15[18:])
