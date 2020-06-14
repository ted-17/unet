# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:11:10 2020

@author: Ted
"""
import numpy as np
import librosa
import glob
from os.path import basename, splitext
#%% utility methods
def get_wavlist(dir):
    """
    find .wav file and make wavlist from the specified dir
    """
    get_list = glob.glob(dir+'/*.wav')
    return get_list

def make_spectrogram(pathname, fftsize, hopsize, nbit, istest=False):
    """
    make spectrogram from pathname
    making it easy for network to train data, compress-normalize with "nbit"
    """
    X = librosa.core.stft(x, n_fft=fftsize, hop_length=hopsize)
    absX = np.abs(X)
    phsX = np.exp(1.j*np.angle(X))
    maxval, minval = np.max(absX), np.min(absX)
    absXn_int= np.floor(((absX-minval)/(maxval-minval))*(2**nbit-1) + 0.5)
    absXn = absXn_int/(2**nbit-1) #0-1
    if istest:
        return absXn, phsX, maxval, minval
    else:
        return absXn, phsX

def mix_voice_noise(voicedir, noisedir, mixeddir, num_data, fs=16000):
    """
    from eachpath, mix voice and noise signal and save into mixedpath
    input:
        wavlist_voice: voice list
        wavlist_noise: noise list
        num_data: number of data to generate
    return:
        x_list: list of mixed signal
        v_list: list of voice signal
        each index must NOT be changed
    """
    wavlist_voice = get_wavlist(voicedir)
    wavlist_noise = get_wavlist(noisedir)
    for i in range(num_data):
        v = librosa.core.load(wavlist_voice[i],sr=fs)
        n = librosa.core.load(wavlist_noise[i],sr=fs)
        filename_voice = splitext(basename(wavlist_voice[i]))[0]
        savepath = mixeddir+'/'+filename_voice+'_mixed.wav'
        x = v + n
        librosa.output.write_wav(savepath,x,fs)

def make_dataset(wavlist, fftsize, hopsize, nbit):
    """
    find .wav file and make spectrogram
    """
    Xbox = []
    for path in wavlist:
        X,_ = make_spectrogram(path, fftsize, hopsize, nbit)
        Xbox.append(X)
    return np.array(Xbox)[...,np.newaxis]