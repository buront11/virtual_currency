import torch
import torch.nn as nn

import joblib

import model
import utils
from config import *

def load_dataset(args):
    # TODO データセット読み出し
    data_types = args.type if args.type is not None else data_types
    datasets = []
    for data_type in data_types:
        datasets.append(joblib.load('./data/'+data_type+'.pkl.cmp'))

    return datasets
    

def train(args):
    # TODO 訓練プログラム
    datasets = load_dataset(args)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',nargs='*')
    args = parser.parse_args()