import csv
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

def split_df_date_times():
    pass

def date_data_plot(csv_path):
    df = pd.read_csv(csv_path)

    # df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m")

    # columsの先頭は必ずDateなのでそれ以外をloop
    colums = df.columns[1:]

    for colum in colums:
        # 描画
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["Date"], df[colum])

        locator = mdates.MonthLocator(bymonthday=None, interval=6, tz=None)
        # 軸目盛の設定
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ## 補助目盛りを使いたい場合や時刻まで表示したい場合は以下を調整して使用
        # ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1), tz=None))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))

        # 軸目盛ラベルの回転
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=0, fontsize=10);
        plt.savefig('./plot/'+colum+'.png')
        print('finished {}'.format(colum))

        ax.grid()
        plt.clf()
        plt.close('all')


