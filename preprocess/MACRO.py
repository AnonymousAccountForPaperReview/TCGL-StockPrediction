# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

dataset = 's&p500' # csi300 or s&p500
raw_path = f"datasets/{dataset}/"  
path = f"immediate_data/{dataset}/" # to save results


def file_name():
    with open(raw_path + "stock_symbols.csv", "r", encoding="utf-8") as fr:
        stock_symbols = fr.readlines()
        stock_symbols = [sy.strip() + ".csv" for sy in stock_symbols]
    return stock_symbols


files = file_name()

symbol_features = {}
for file in files:
    features_df = pd.read_csv(path + "features/" + file)
    symbol_features[file] = features_df
date_15 = pd.read_csv(path + "features/AAPL.csv")["Date"]

Macro_data = {}
for date in date_15:
    Macro_data[date] = []
for key, value in symbol_features.items():
    dates = symbol_features[key].Date
    close = symbol_features[key].Close
    for i in range(len(date_15)):
        Macro_data[date_15[i]].append(close[i])

ret = pd.DataFrame(index=["close"], columns=date_15)
for key, value in Macro_data.items():
    tmp = Macro_data[key]
    avg_close = np.mean(np.array(tmp))
    std_close = np.std(np.array(tmp))
    skew_close = np.mean((tmp - avg_close) ** 3)
    kurt_close = np.mean((tmp - avg_close) ** 4) / (std_close**4)
    ret.loc["close", key] = avg_close

ret.to_csv(f"{path}/Macro_data.csv")
