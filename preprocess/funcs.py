import os
import argparse
import torch
import pandas as pd
import numpy as np
from scipy import integrate
import warnings
import math

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

args = parser.parse_args()


def file_name(raw_path):
    with open(raw_path + "stock_symbols.csv", "r", encoding="utf-8") as fr:
        stock_symbols = fr.readlines()
        stock_symbols = [sy.strip() + ".csv" for sy in stock_symbols]
    return stock_symbols


def preprocess(data):
    if data == "s&p500":
        raw_path = "datasets/s&p500/"
    elif data == "csi300":
        raw_path = "datasets/csi300/"
    else:
        print("Invalid dataset!")
        return

    trade_data = pd.read_csv(raw_path + "date.csv")
    trade_data.set_index("Date", inplace=True)
    files = file_name(raw_path)

    num_of_days = pd.read_csv(raw_path + "raw_data/" + files[0]).shape[0]
    num_stocks = len(files)
    name_of_stocks = (
        pd.read_csv(raw_path + "stock_symbols.csv", header=None).values[:, 0].tolist()
    )
    # return_rate = pd.DataFrame(np.zeros((num_of_days,num_stocks)))
    # return_rate.columns = files
    close_price = pd.DataFrame(np.zeros((num_of_days, num_stocks)))
    close_price.columns = files

    for file in files:
        pre_stock = pd.read_csv(raw_path + "raw_data/" + file)
        for i in range(num_of_days):
            close_price[file][i] = pre_stock["Close"][i]
        # if i == 0:
        #     return_rate[file][i] = np.nan
        # else:
        #     return_rate[file][i] = pre_stock['Close'][i] / pre_stock['Close'][i-1]

    close_price.columns = name_of_stocks
    # return_rate = return_rate.dropna(axis=0).reset_index().drop('index', axis=1)
    return close_price


def attractor(series, E, tau):
    L = len(series)
    att = np.zeros((L, E))
    for i in range((E - 1) * tau, L):
        for j in range(E):
            att[i, j] = series[i - j * tau]
    att = att[(E - 1) * tau :,]
    return att


def Manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2))


def Euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def integrand(aa):
    return np.exp(-(aa**2))


def pc_pattern(effect, cause, L2, E):
    if E >= 2:
        aa = np.sum(np.abs(effect[0:L2,]), axis=1) / np.sum(
            np.abs(cause[0:L2,]), axis=1
        )
        pc = []
        for i in range(L2):
            integral = integrate.quad(
                integrand, -aa[i], aa[i]
            )  # ensure the integral result is positive
            # if (np.isnan(integral) == True):
            #     print(aa[i])
            pc.append(integral[0] / (np.pi**0.5))
    return pc


def dark(X, Y, E, tau, method, threshold, horizon):
    if horizon > 0:
        X = X[:-horizon]
        Y = Y[horizon:]

    Mx = attractor(X, E, tau)
    My = attractor(Y, E, tau)

    L1 = Mx.shape[0]

    xx = Mx.T.reshape((Mx.size, 1)).repeat(L1, axis=1)
    yy = My.T.reshape((My.size, 1)).repeat(L1, axis=1)

    xx0 = xx[0:L1,] - xx[0:L1,].T
    xx1 = xx[L1 : 2 * L1,] - xx[L1 : 2 * L1,].T
    yy0 = yy[0:L1,] - yy[0:L1,].T
    yy1 = yy[L1 : 2 * L1,] - yy[L1 : 2 * L1,].T

    if method == "Manhattan":
        Dx = np.abs(xx0) + np.abs(xx1)
        Dy = np.abs(yy0) + np.abs(yy1)

        if E > 2:
            for i in range(2, E):
                xx_i = xx[i * L1 : (i + 1) * L1,] - xx[i * L1 : (i + 1) * L1,].T
                yy_i = yy[i * L1 : (i + 1) * L1,] - yy[i * L1 : (i + 1) * L1,].T
                Dx = Dx + np.abs(xx_i)
                Dy = Dy + np.abs(yy_i)

    elif method == "Euclidean":
        Dx = (xx0**2 + xx1**2) ** 0.5
        Dy = (yy0**2 + yy1**2) ** 0.5
        if E > 2:
            for i in range(2, E):
                xx_i = xx[i * L1 : (i + 1) * L1,] - xx[i * L1 : (i + 1) * L1,].T
                yy_i = yy[i * L1 : (i + 1) * L1,] - yy[i * L1 : (i + 1) * L1,].T
                Dx = (Dx**2 + xx_i**2) ** 0.5
                Dy = (Dy**2 + yy_i**2) ** 0.5
    else:
        print("Invalid method!")

    # Find the E+1 nearest neighbors
    findx_nn = np.zeros((L1, E + 1))
    findy_nn = np.zeros((L1, E + 1))

    for i in range(L1):
        findx_nn[i,] = np.argsort(Dx[i,])[1 : (E + 2)]
        findy_nn[i,] = np.argsort(Dy[i,])[1 : (E + 2)]

    findx_nn = findx_nn.astype(int)
    findy_nn = findy_nn.astype(int)

    Wx = np.zeros((L1, E + 1))
    Wy = np.zeros((L1, E + 1))

    for i in range(L1):
        Wx[i,] = np.exp(-Dx[i, findx_nn[i,]])
        Wy[i,] = np.exp(-Dy[i, findy_nn[i,]])
        Wx[i,] = Wx[i,] / np.sum(Wx[i,])
        Wy[i,] = Wy[i,] / np.sum(Wy[i,])

    sx = np.zeros((L1, E - 1))
    sy = np.zeros((L1, E - 1))

    for i in range(E - 1):
        sx[:, i] = (Mx[:, i + 1] - Mx[:, i]) / Mx[:, i]
        sy[:, i] = (My[:, i + 1] - My[:, i]) / My[:, i]

    Sx = np.zeros((L1, E - 1))
    Sy = np.zeros((L1, E - 1))

    for i in range(L1):
        for j in range(E - 1):
            Sx[i, j] = np.sum(sx[findx_nn[i,], j] * Wx[i,])
            Sy[i, j] = np.sum(sy[findy_nn[i,], j] * Wy[i,])

    pc_x2y = pc_pattern(Sy, Sx, L1, E)

    PC_x2y = np.zeros((3 ** (E - 1), 3 ** (E - 1)))

    if E == 2:
        bs = ["d", "h", "u"]
    elif E == 3:
        bs = ["dd", "hd", "ud", "dh", "hh", "uh", "du", "hu", "uu"]
    elif E == 4:
        bs = [
            "ddd",
            "hdd",
            "udd",
            "dhd",
            "hhd",
            "uhd",
            "dud",
            "hud",
            "uud",
            "ddh",
            "hdh",
            "udh",
            "dhh",
            "hhh",
            "uhh",
            "duh",
            "huh",
            "uuh",
            "ddu",
            "hdu",
            "udu",
            "dhu",
            "hhu",
            "uhu",
            "duu",
            "huu",
            "uuu",
        ]
    elif E == 5:
        bs = [
            "dddd",
            "hddd",
            "uddd",
            "dhdd",
            "hhdd",
            "uhdd",
            "dudd",
            "hudd",
            "uudd",
            "ddhd",
            "hdhd",
            "udhd",
            "dhhd",
            "hhhd",
            "uhhd",
            "duhd",
            "huhd",
            "uuhd",
            "ddud",
            "hdud",
            "udud",
            "dhud",
            "hhud",
            "uhud",
            "duud",
            "huud",
            "uuud",
            "dddh",
            "hddh",
            "uddh",
            "dhdh",
            "hhdh",
            "uhdh",
            "dudh",
            "hudh",
            "uudh",
            "ddhh",
            "hdhh",
            "udhh",
            "dhhh",
            "hhhh",
            "uhhh",
            "duhh",
            "huhh",
            "uuhh",
            "dduh",
            "hduh",
            "uduh",
            "dhuh",
            "hhuh",
            "uhuh",
            "duuh",
            "huuh",
            "uuuh",
            "dddu",
            "hddu",
            "uddu",
            "dhdu",
            "hhdu",
            "uhdu",
            "dudu",
            "hudu",
            "uudu",
            "ddhu",
            "hdhu",
            "udhu",
            "dhhu",
            "hhhu",
            "uhhu",
            "duhu",
            "huhu",
            "uuhu",
            "dduu",
            "hduu",
            "uduu",
            "dhuu",
            "hhuu",
            "uhuu",
            "duuu",
            "huuu",
            "uuuu",
        ]

    PC_x2y = pd.DataFrame(PC_x2y, columns=bs, index=bs)

    pat1 = np.zeros((L1, E - 1)).astype(str)
    pat2 = np.zeros((L1, E - 1)).astype(str)
    for i in range(pat1.shape[0]):
        for j in range(pat1.shape[1]):
            if Sx[i, j] < -threshold:
                pat1[i, j] = "d"
            elif Sx[i, j] >= -threshold and Sx[i, j] <= threshold:
                pat1[i, j] = "h"
            else:
                pat1[i, j] = "u"
    for i in range(pat2.shape[0]):
        for j in range(pat2.shape[1]):
            if Sy[i, j] < -threshold:
                pat2[i, j] = "d"
            elif Sy[i, j] >= -threshold and Sy[i, j] <= threshold:
                pat2[i, j] = "h"
            else:
                pat2[i, j] = "u"

    for i in range(L1):
        find1 = ""
        find2 = ""
        for j in range(E - 1):
            find1 = find1 + pat1[i, j]
            find2 = find2 + pat2[i, j]
        PC_x2y.loc[find1, find2] = PC_x2y.loc[find1, find2] + pc_x2y[i]

    arr_PC_x2y = np.array(PC_x2y)

    return arr_PC_x2y


from multiprocessing import Pool


def causal(
    data,
    E=3,
    tau=1,
    method="Manhattan",
    cau_window=30,
    cau_len=30,
    threshold=0,
    horizon=9,
):
    th = str(threshold).replace(".", "-")

    os.makedirs(f"immediate_data/{args.data}/h{horizon}", exist_ok=1)

    path = f"immediate_data/{args.data}/h{horizon}"

    close_price = preprocess(data)

    total_days = close_price.shape[0]
    total_stocks = close_price.shape[1]
    assert total_days >= cau_len, "Lack of data!"

    num_cal = 1 + int(np.ceil((total_days - cau_len) / cau_window))

    print(num_cal)

    # causal[k,i,j] refers to the k-th window, causality of stock i to stock j
    causal_matrices = np.zeros(
        (3 ** (E - 1), 3 ** (E - 1), num_cal, total_stocks, total_stocks)
    )
    np.save(path + f"/h{horizon}.npy", causal_matrices)
    dir = path + f"/h{horizon}.npy"

    pool = Pool(processes=20)

    for i in range(total_stocks):
        print("Stock {}".format(i))
        stockX = close_price.iloc[:, i]
        for j in range(i, total_stocks):
            stockY = close_price.iloc[:, j]
            for k in range(num_cal):
                pool.apply_async(
                    func,
                    args=(
                        stockX,
                        stockY,
                        i,
                        j,
                        k,
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
                    # callback=mycallback,
                )
    pool.close()
    pool.join()

    print(
        "emdim={}, lag={}, method={}, window={}, nday={}, threshold={}, horizon={}".format(
            E, tau, method, cau_window, cau_len, threshold, horizon
        )
    )
    return causal_matrices


def mycallback(ret):
    x, y, i, j, k, dir = ret
    causal_matrices = np.load(dir)
    causal_matrices[:, :, k, i, j] = x
    causal_matrices[:, :, k, j, i] = y
    np.save(dir, causal_matrices)


def func(
    stockX: pd.DataFrame,
    stockY: pd.DataFrame,
    i: int,
    j: int,
    k: int,
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
    begin = k * cau_window
    end = begin + cau_len
    # if total_days - begin <= E * tau + E:
    if total_days - begin < cau_len:
        begin = total_days - cau_len
    Xprice = stockX[begin:end].tolist()
    Yprice = stockY[begin:end].tolist()

    arr_PC_x2y = dark(Xprice, Yprice, E, tau, method, threshold, horizon)
    arr_PC_y2x = dark(Yprice, Xprice, E, tau, method, threshold, horizon)
    return (arr_PC_x2y, arr_PC_y2x, i, j, k, dir)


if __name__ == "__main__":
    causal(
        data=args.data,
        E=args.emdim,
        tau=args.lag,
        method=args.method,
        cau_window=args.window,
        cau_len=args.nday,
        threshold=args.threshold,
        horizon=args.horizon,
    )
