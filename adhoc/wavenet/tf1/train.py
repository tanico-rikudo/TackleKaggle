import os
import argparse
import librosa
import random
import numpy as np

import audio_reader as ar
import model as m

# define parser and command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-d',help='input root directory path of speech data',required=True)
args = parser.parse_args()

# preprocess parameter
sample_rate = 16000
n_fft=1024
hop_length=256
n_mels=128
threshold=20
receptive_field=7680
input_quantize=256
directory=args.d

# training parameter
#frame_size = 2048 #size of one frame audio
output_quantize = 256
r_channels = 64
a_channels = 256
s_channels = 256
d_channels = r_channels*2

n_layer = 10
n_loop = 4

n_epochs = 2000


# create iterator after read data and preprocess data
iterator = ar.AudioReader(sample_rate, n_fft, hop_length, n_mels,
				threshold, receptive_field, input_quantize, directory=directory)
iterator = iterator.preprocess()

# create model
model_wavenet = m.model(receptive_field, r_channels, s_channels,
				a_channels, d_channels, n_layer, n_loop)
model_wavenet.wavenet()