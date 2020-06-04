import tensorflow as tf
import numpy as np
import os
from utils import *

# parameters
import hparams
from wavenet.utils import files_to_list, mulaw_quantize, trim_silence, melspectrogram, load_wav


### tf.trainで許可されている型 へ変換する
# value=(配列の形)
# tf.train.Int64List
# tf.train.FloatList
# tf.train.BytesList
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float32_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def audio_preprocess(wav_path):
    wav = load_wav(wav_path, sampling_rate = hparams.sampling_rate)
    wav = trim_silence(wav, top_db=40, fft_size=2048, hop_size=512)
    wav = normalize(wav) * 0.95 #[-0.95, 0,95]へ
    mel_sp = melspectrogram(
        wav,
        sampling_rate=hparams.sampling_rate,
        num_mels=hparams.num_mels,
        n_fft=hparams.n_fft,
        hop_size=hparams.hop_size,
        window_size=hparams.window_size)

    # hop_sizeの整数倍になるように最後尾をpadする
    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    pad = (wav.shape[0] // hparams.hop_size + 1) * hparams.hop_size - len(wav)
    wav = np.pad(wav, (0, pad), mode='constant', constant_values=0.0)
    assert len(wav) % hparams.hop_size == 0

    # 量子化
    wav = mulaw_quantize(wav, 255)

    # mel特徴量の平坦化
    mel_sp_channels, mel_sp_frames = mel_sp.shape
    mel_sp = mel_sp.flatten()

    # tf.train.Exampleクラスはデータ１件分に相当します。
    # 引数に保存したいデータをtf.train.Featuresクラスにて指定します。
    # tf.train.Featuresクラスは、
    # 引数に保存したいデータをKey - Value型で指定します。
    # その際、Valueで指定する、tf.train.Featureクラスの引数は３種類のみ
    record = tf.train.Example(
        features=tf.train.Features(
            feature={
                'wav': _int64_array_feature(wav),
                'mel_sp': _float32_array_feature(mel_sp),
                'mel_sp_frames': _int64_feature(mel_sp_frames),
    }))

    return record


def createTFRecord():
    os.makedirs(hparams.result_dir, exist_ok=True)

    train_files = files_to_list(hparams.train_files)
    # 一つのTFrecordにすべてを書き込み
    with tf.io.TFRecordWriter(hparams.result_dir + "train_data.tfrecord") as writer:
        for wav_path in train_files:
            record = audio_preprocess(wav_path)
            writer.write(record.SerializeToString())


if __name__ == '__main__':
    # 事前に対象を作成
    # ls wavs/*.wav | tail -n+10 > train_files.txt
    # ls wavs/*.wav | head -n-10 > test_files.txt
    createTFRecord()