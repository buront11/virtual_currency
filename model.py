import torch
import torch.nn as nn
from config import *

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100,
                 output_size=1, batch_size = 32):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, self.batch_size, self.hidden_layer_size),
                            torch.zeros(1, self.batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(seq_len, batch_size, 1),
                                               self.hidden_cell) #lstmのデフォルトの入力サイズは(シーケンスサイズ、バッチサイズ、特徴量次元数)
        predictions = self.linear(self.hidden_cell[0].view(batch_size, -1))
        return predictions[:, 0]


class Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1):
        '''
        Args:
        -----
        input_dim   :   int
                        入力シーケンスデータの次元
        hidden_dim  :   int
                        LSTMの隠れ層の次元
        output_dim  :   int
                        出力シーケンスデータの次元
        '''
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, sequence, hidden0=None):
        '''
        Args:
        -----
        sequence    :   multi-dimensions list
                        Encoder入力するシーケンスデータ
        hidden0     :   tuple
                        隠れ層とセルの初期状態を意味するタプル

        Returns:
        --------
        state       :   tuple
                        LSTMから最終的に出力される隠れ層とセル
        '''
        # Many to Oneなので、第２戻り値を使う
        output, state = self.lstm(sequence, hidden0)
        # state = (h, c)
        return state


class Decoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # LSTMのhidden_dim次元の隠れ層をoutput_dim次元に変換する全結合層
        self.hidden2linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence, encoder_state):
        # Many to Manyなので、第１戻り値を使う。
        # 第２戻り値は推論時に次の文字を生成するときに使います。
        output, state = self.lstm(sequence, encoder_state)
        output = self.hidden2linear(output)
        return output, state
