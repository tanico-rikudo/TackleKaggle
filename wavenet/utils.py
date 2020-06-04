import numpy as np
import librosa
from scipy.io import wavfile
import hparams
# OK
"""
ファイル入出力
"""
def files_to_list(filepath):
    """
    :param filepath:
    :return filelist:
    """
    with open(filepath, encoding="utf-8") as f:
        ls_files = f.readlines()
    ls_files = [ _file.rstrip() for _file in ls_files]
    return ls_files


def load_wav(filepath, sampling_rate):
    wav = librosa.core.load(filepath, sr=sampling_rate)[0]
    return wav


def save_wav(wav, savepath, sampling_rate):
    # 保存
    wav *= 32767 / max(0.0001, np.max(np.abs(wav)))
    wavfile.write(savepath, sampling_rate, wav.astype(np.int16))

"""
特徴
"""
def trim_silence(wav, top_db=40, fft_size=2048, hop_size=512):
    # 無音区間の除去
    # https://qiita.com/kshina76/items/5686923dee2889beba7c
    # top_db以下は削除
    # hop_length: The number of samples between analysis frames
    return librosa.effects.trim(wav, top_db=top_db, frame_length=fft_size, hop_length=hop_size)[0]


def normalize(wav):
    # 正規化
    return librosa.util.normalize(wav)


def mulaw_core(x, mu=255):
    # mu-law計算
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def mulaw_quantize(x, mu=255):
    # mu-law量子化
    x = mulaw_core(x)
    x = (x + 1) / 2 * mu
    return x.astype(np.int)


def inv_mulaw_core(x, mu=255):
    # mu-law計算(原系列へ)
    return np.sign(x) * (1.0 / mu) * ((1.0 + mu) ** np.abs(x) - 1.0)


def inv_mulaw_quantize(x, mu=255):
    # mu-law量子化(原系列へ)
    x = 2 * x.astype(np.float32) / mu - 1
    return inv_mulaw_core(x, mu)


def melspectrogram(wav, sampling_rate, num_mels, n_fft, hop_size, window_size):
    # メル特徴量計算
    # https://tips-memo.com/python-logmel#i
    # wavファイル（生の音）にSTFT（短時間フーリエ変換）を施して
    # メルフィルタバンクを適用した特徴量
    # 短時間フーリエ変換では，音の周波数に関する時間変化を表すスペクトログラムという特徴量を得ることができます。
    # そのスペクトログラムを，人間の聴覚特性にフィットするような形に整形（＝メルフィルタバンクを適用する）
    # したものがメル周波数スぺクトログラムです
    # Output -> (timeframe, mel_dim)

    # 窓関数：window_size (= n_fftとなる)
    # hop_size:窓関数のスライド幅
    d = librosa.stft(
        y=wav,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=window_size,
        pad_mode='constant')

    mel_filter = librosa.filters.mel(
        sampling_rate,
        n_fft,
        n_mels=num_mels)
    s = np.dot(mel_filter, np.abs(d))

    return np.log10(np.maximum(s, 1e-5))


if __name__ == '__main__':
    wav = load_wav("./wavs/p376_001.wav",hparams.sampling_rate)
    wav = trim_silence(wav, top_db=40, fft_size=2048, hop_size=512)
    wav = normalize(wav) * 0.95 #[-0.95, 0,95]へ
    mel_sp = melspectrogram(
        wav,
        sampling_rate=hparams.sampling_rate,
        num_mels=hparams.num_mels,
        n_fft=hparams.n_fft,
        hop_size=hparams.hop_size,
        window_size=hparams.window_size)
    print ()