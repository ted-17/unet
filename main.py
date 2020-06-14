# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:11:10 2020

@author: Ted
"""
#%%
import os
import librosa
import numpy as np
from keras.layers import Input,Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Activation, BatchNormalization, Reshape
from keras.models import Model
from sklearn.model_selection import train_test_split
import time

import util
from network import UNet

#%% make wavlist
voicedir = "voice/wav"
noisedir = "noise/wav"
mixeddir = "mixed/wav"
fftsize, hopsize, nbit = 512, 256, 8
num_data = 100

# mix voice and noise and save it as .wav file
util.mix_voice_noise(voicedir, noisedir, mixeddir, num_data, fs=16000)

# get list each of which the path name is written
voicepath_list = util.get_wavlist(voicedir)
mixedpath_list = util.get_wavlist(mixeddir)

# make spectrogram (n x F x T x 1)
V = util.make_dataset(voicepath_list, fftsize, hopsize, nbit)
X = util.make_dataset(mixedpath_list, fftsize, hopsize, nbit)

#%% model training
height, width = fftsize//2, fftsize//2 #CNN height x width
X_train = X[:,:height,:width,...] #voice + noise
Y_train = V[:,:height,:width,...] #noise only
num_filt_first = 16
unet = UNet(height, width, num_filt_first)
model = unet.get_model()
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, Y_train, epochs=5, batch_size=32)

#%% model testing
absY, phsY, max_Y, min_Y = util.make_spectrogram(mixpath_list[0], fftsize, hopsize, nbit, istest=True)
P = np.squeeze(model.predict(absY[np.newaxis,:height,:width,...]))
P = np.hstack((P,absY[:height,width:])) #t-axis
P = np.vstack((P,absY[height,:])) #f-axis
Y = (absY*(max_Y-min_Y)+min_Y)*phsY
y = librosa.core.istft(absY*phsY, hop_length=hopsize, win_length=fftsize)