import tensorflow as tf
import numpy as np

from .module import Conv1D, ReLU, ResidualConv1DGLU
from .upsample import UpsampleNetwork
from utils import mulaw_quantize


class WaveNet(tf.keras.Model):
    def __init__(self, num_mels, upsample_scales):
        super().__init__()
        self.upsample_network = UpsampleNetwork(upsample_scales)

        # 冒頭のcausalブロック
        # padding:"causal"はcausal（dilated）畳み込み．
        # 例えば，output[t]はinput[t + 1]に依存しません．
        # 時間的順序を無視すべきでない時系列データをモデリングする際に有効です．(WaveNet)
        self.first_layer = Conv1D(128,
                                  kernel_size=1,
                                  padding='causal')

        # Residual Block(コレの積み重ね)
        # 層が深くなるにつれて畳み込むユニットをスキップ（離す( = Dilation)）
        # ２倍はなに？
        self.residual_blocks = []
        for _ in range(2):
            for i in range(10):
                self.residual_blocks.append(
                    ResidualConv1DGLU(128,
                                      256,
                                      kernel_size=3,
                                      skip_out_channels=128,
                                      dilation_rate=2 ** i)
                )

        # 最後のまとめ層
        # 連続して入出力を実施する
        self.final_layers = [
            ReLU(),
            Conv1D(128,
                   kernel_size=1,
                   padding='causal'),
            ReLU(),
            Conv1D(256,
                   kernel_size=1,
                   padding='causal')
        ]

    @tf.function
    def call(self, inputs, c):
        """
        :param inputs:wave
        :param c:mel
        :return:
        """
        #
        c = tf.expand_dims(c, axis=-1)
        c = self.upsample_network(c)

        # サイズが１の次元を消去＆入れ替え
        # https://qiita.com/cfiken/items/04925d4da39e1a24114e#tfsqueeze
        c = tf.transpose(tf.squeeze(c, axis=-1), perm=[0, 2, 1])

        # 最初のレイヤー：causal_conv
        x = self.first_layer(inputs)

        # 途中のk-layers: ResidualNet
        skip_out = None
        for block in self.residual_blocks:
            x, h = block(x, c)
            if skip_out is None:
                skip_out = h
            else:
                skip_out = skip_out + h

        # 最後のまとめ層：kip-connections
        x = skip_out
        for layer in self.final_layers:
            x = layer(x)

        return x
