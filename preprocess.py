# TODO trainデータをとりあえず３日ごとに分割する
import pandas as pd
import numpy
import joblib

from config import *

def preprocess(args):
    
    params = args.type if args.type is not None else data_types

    for param in params:
        train_df = pd.read_csv('./data/train.csv')
        train_df.std(ddof=0)
        dataset = []
        tmp_list = []
        for df_date,df_data in zip(train_df['Date'],train_df[param+'_Close_log']):
            tmp_list.append([df_date,df_data])
            # 60分x24時間x３日で4320
            if len(tmp_list) == 4320:
                dataset.append(tmp_list)
                tmp_list = []

        joblib.dump(dataset, './data/'+param+'.pkl.cmp', compress=True)
                
        
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',nargs='*')
    args = parser.parse_args()
    preprocess(args)