# coding: utf-8
import multiprocessing
import time
from funcs import causal, preprocess, dark
import os
import argparse
import torch
import pandas as pd
import numpy as np
from scipy import integrate
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Calculation of Causalities with Horizon")

parser.add_argument(
    "--data", type=str, default="s&p500", help="name of the target dataset"
)
parser.add_argument("--emdim", type=int, default=3, help="embedding parameter")
parser.add_argument("--lag", type=int, default=1, help="embedding parameter")
parser.add_argument(
    "--method",
    type=str,
    default="Manhattan",
    help="method for calculating distance between attractors",
)
parser.add_argument("--window", type=int, default=30, help="period of calculation")
parser.add_argument(
    "--nday", type=int, default=30, help="number of days involved in each calculation"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.005,
    help="threshold for definition of dynamic characteristics",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=9,
    help="horizon of time series for calculation of causalities",
)


def mycallback(ret):
    x, y, i, j, dir = ret
    time1 = time.time()
    print(f"calling_back for {i},{j}")
    causal_matrices = np.load(dir)
    causal_matrices[:, :, :, i, j] = x
    causal_matrices[:, :, :, j, i] = y
    np.save(dir, causal_matrices)
    time2 = time.time()
    print(f"used {time2-time1} seconds")
    print("back_to_mainstream")


def func(
    mg_list,
    stockX: pd.DataFrame,
    stockY: pd.DataFrame,
    i: int,
    j: int,
    total_stocks: int,
    close_price: pd.DataFrame,
    E: int,
    tau: int,
    method: str,
    threshold: float,
    horizon: int,
    num_cal: int,
    cau_window: int,
    cau_len: int,
    total_days: int,
    dir: str,
):
    tmp1 = np.zeros((3 ** (E - 1), 3 ** (E - 1), num_cal))
    tmp2 = np.zeros((3 ** (E - 1), 3 ** (E - 1), num_cal))

    for k in range(num_cal):
        begin = k * cau_window
        end = begin + cau_len
        # if total_days - begin <= E * tau + E:
        if total_days - begin < cau_len:
            begin = total_days - cau_len
        Xprice = stockX[begin:end].tolist()
        Yprice = stockY[begin:end].tolist()

        arr_PC_x2y = dark(Xprice, Yprice, E, tau, method, threshold, horizon)
        arr_PC_y2x = dark(Yprice, Xprice, E, tau, method, threshold, horizon)

        tmp1[:, :, k] = arr_PC_x2y
        tmp2[:, :, k] = arr_PC_y2x

        # print(tmp1.sum(), tmp2.sum())
    # print(f"fill in for {i}, {j}")
    mg_list.append([tmp1, tmp2, i, j])
    return


if __name__ == "__main__":
    args = parser.parse_args()
    data = args.data
    E = args.emdim
    tau = args.lag
    method = args.method
    cau_window = args.window
    cau_len = args.nday
    threshold = args.threshold
    horizon = args.horizon
    th = str(threshold).replace(".", "-")

    os.makedirs(f"immediate_data/{args.data}/h{horizon}", exist_ok=1)

    path = f"immediate_data/{args.data}"

    close_price = preprocess(data)

    total_days = close_price.shape[0]
    total_stocks = close_price.shape[1]
    assert total_days >= cau_len, "Lack of data!"

    num_cal = 1 + int(np.ceil((total_days - cau_len) / cau_window))

    print(num_cal)

    # causal[k,i,j] refers to the k-th window, causality of stock i to stock j

    manager = multiprocessing.Manager()
    mg_list = manager.list()
    pool = multiprocessing.Pool(processes=20)

    for i in range(total_stocks):
        stockX = close_price.iloc[:, i]
        for j in range(i, total_stocks):
            stockY = close_price.iloc[:, j]
            pool.apply_async(
                func=func,
                args=(
                    mg_list,
                    stockX,
                    stockY,
                    i,
                    j,
                    total_stocks,
                    close_price,
                    E,
                    tau,
                    method,
                    threshold,
                    horizon,
                    num_cal,
                    cau_window,
                    cau_len,
                    total_days,
                    dir,
                ),
            )
    pool.close()
    pool.join()
    print(len(mg_list))

    causal_matrices = np.zeros(
        (3 ** (E - 1), 3 ** (E - 1), num_cal, total_stocks, total_stocks)
    )
    for item in mg_list:
        causal_matrices[:, :, :, item[2], item[3]] = item[0]
        causal_matrices[:, :, :, item[3], item[2]] = item[1]

    np.save(path + f"/causal/h{horizon}/h{horizon}.npy", causal_matrices)
    print(
        "emdim={}, lag={}, method={}, window={}, nday={}, threshold={}, horizon={}".format(
            E, tau, method, cau_window, cau_len, threshold, horizon
        )
    )