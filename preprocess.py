# TODO trainデータをとりあえず３日ごとに分割する
import pandas as pd
import numpy
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import utils
from config import *

def preprocess(args):
    column_nums = []
    if args.close:
        # close_logに対応するcolumn番号を追加
        column_nums.extend([8,12,16,20])

    scaler = MinMaxScaler(feature_range=(-1, 1))

    df = pd.read_csv(args.path)
    for column_num in column_nums:
        df = df[df_columns[column_num]]
        all_data = df.values.astype(float)

        # 予測問題なのでtestデータは最後の日にちになるようにランダムには取らない
        train_data = all_data[:-times]
        test_data = all_data[-times:]

        train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
        train_inout_seq = create_inout_sequences(input_data, times)

        joblib.dump(dataset, './data/'+df_columns[column_num]+'.pkl.cmp', compress=True)
                
def create_inout_sequences(input_data, sl):
    """trainに対応するシーケンスデータに対応するラベルを作成する関数

    Parameters
    ----------
    input_data : numpy.array
        trainデータとなるシーケンスデータ
    sl : int
        sequence length:シーケンスデータ１つ分の長さ

    Returns
    -------
    list
        trainデータのラベル
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
        
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--close',action='store_true')
    parser.add_argument('-p','--path',required=True)
    args = parser.parse_args()
    preprocess(args)