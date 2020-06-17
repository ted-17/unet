# unet
noise reduction using deep unet

<img width="602" alt="unet" src="https://user-images.githubusercontent.com/66860222/84907987-d7f3b700-b0ee-11ea-993a-0fa553a5c27f.png">
Input Noisy Spectrogram, and output Clean Spectrogram corresponding to the input.<br>
Similar to Convolutional Auto Encoder, input and output are evaluated by Mean Squared Error to be minimized.

Unlike AutoEncoder, Unet has contracting and expansive paths which enables network to predict a good segmentation map.
