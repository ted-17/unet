# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:11:10 2020

@author: Ted
"""

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.layers.merge import concatenate
from keras.layers import Activation, BatchNormalization, Dropout

class UNet(object):
    def __init__(self, height=256, width=128,num_filt_first=16):
        self.FILT_SIZE = 4
        self.STRIDE_SIZE = 2

        #input
        inputs = Input((height, width, 1))

        # create encoder
        enc1 = Conv2D(num_filt_first, self.FILT_SIZE, strides=self.STRIDE_SIZE,padding='same')(inputs) #(1->16)
        enc2 = self.add_enc(num_filt_first*2, enc1) #(16->32)
        enc3 = self.add_enc(num_filt_first*4, enc2) #(32->64)
        enc4 = self.add_enc(num_filt_first*8, enc3) #(64->128)
        enc5 = self.add_enc(num_filt_first*16, enc4) #(128->256)
        enc6 = self.add_enc(num_filt_first*32, enc5) #(256->512)

        # create decoder
        dec1 = self.add_dec(num_filt_first*16, enc6, dropout=True) #(512->256)
        dec1 = concatenate([dec1, enc5],axis=-1) #[256,256]->512
        dec2 = self.add_dec(num_filt_first*8, dec1, dropout=True) #(512->128)
        dec2 = concatenate([dec2, enc4]) #[128, 128]->256
        dec3 = self.add_dec(num_filt_first*4, dec2, dropout=True) #(256->64)
        dec3 = concatenate([dec3, enc3]) #[64, 64]->128
        dec4 = self.add_dec(num_filt_first*2, dec3, dropout=False) #(128->32)
        dec4 = concatenate([dec4, enc2]) #[32, 32]->64
        dec5 = self.add_dec(num_filt_first*1, dec4, dropout=False) #(64->16)
        dec5 = concatenate([dec5, enc1]) #[16, 16]->32
        out = self.add_dec_final(dec5) #(32->1)
        self.UNET = Model(input=inputs, output=out)

    def add_enc(self, filter_count, sequence):
        new_sequence = Conv2D(filter_count, self.FILT_SIZE, strides=self.STRIDE_SIZE, padding='same')(sequence)
        new_sequence = BatchNormalization()(new_sequence)
        new_sequence = Activation(activation='relu')(new_sequence)
        return new_sequence

    def add_dec(self, filter_count, sequence, dropout=True):
        sequence = UpSampling2D(size = (2,2))(sequence)
        new_sequence = Conv2D(filter_count, self.FILT_SIZE, strides=1, padding='same',kernel_initializer='he_normal')(sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if dropout: new_sequence = Dropout(0.5)(new_sequence)
        new_sequence = Activation(activation='relu')(new_sequence)
        return new_sequence

    def add_dec_final(self, sequence):
        sequence = UpSampling2D(size = (2,2))(sequence)
        new_sequence = Conv2D(1, self.FILT_SIZE, strides=1, padding='same',kernel_initializer='he_normal')(sequence)
        new_sequence = Activation(activation='sigmoid')(new_sequence)
        return new_sequence

    def get_model(self):
        return self.UNET



