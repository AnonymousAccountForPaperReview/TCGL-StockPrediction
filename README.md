# GAT-TCG

## Introduction
This repository contains the data and code for the paper "Temporal Causal Graph Learning for Stock Prediction". We include everything from dataset to data preprocessing, and of course the final model as well as code for training that you can develop based on your own intention. 

## Datasets
This paper collected datasets from Yahoo Finance for two major stock markets in the US and China. The datasets "s&p500" and "csi300" represent major companies in two market indices: Standard and Poor’s 500 (S&P 500) and China Securities Index 300 (CSI 300). In each dataset folder, there are two types of files for each dataset. The file "./raw_data/stock_name.csv" contains the raw data for each stock during the given period. The raw data include five features: the opening price, high price, low price, closing price, and trading volume.

The file "date.csv" gives the data collection period.

The file "stock_symbols.csv" is the list of name of stocks in the corresponding market index.

## Installation
`pip install -r requirements.txt`

## Example of usage
Here we show how to reproduce the results from the S&P 500 dataset. 

### Step 1: Labeling

For each stock, generate the labels under different tau values.

`python preprocess/labeling.py`

### Step 2: Generate causal matrices for all pairs of stocks
To train a GAT-TCG model, three causality matrices are required by default, which can be reproduced by the following command.

`for h in {0,3,5,14}; do python preprocess/compute_causal.py --horizon $h; done`

This will save results to folder `immediate_data/<dataset>/causal/h*/h*.npy`

Meanwhile, we recommend to directly download them from [Google Drive](https://drive.google.com/drive/folders/1apilzFAu4okxD-Luq90NHBbhlR9ivY4j?usp=sharing), because generating multiple causality matrices may take several hours of time. 

### Step 3: Compute the causal intensity and construct the graph
First, compute the causal intensity by
`python preprocess/causality.py`. 
This will generate results to folder `immediate_data/<dataset>/graph`

Then, construct the causal graphs with
`python preprocess/DSCMG.py`. This will generate results to folder `immediate_data/<dataset>/graph_date`

### Step 4: Generate market-level info.
The file `preprocess/MACRO.py` contains code to preprocess data regarding the temporal dynamics, which contributes to the idea of GAT-TCG. With causality matrices and graphical data prepared under corresponding folders, we can train a GAT-TCG model simply through the following code.

## Step 5: Training

`models/GATTCG.py` contains the code for model GAT-TCG. To start training, just run: 

`python models/train.py --tau 9`

This will save results to folder `experiments`.

The file "utils.py" is for data loading and evaluation metrics.