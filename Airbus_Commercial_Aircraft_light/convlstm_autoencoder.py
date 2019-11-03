import data
from colorama import Fore, Back, Style
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import Model, model_from_json, Sequential
from keras.layers import Input, Conv3D, ConvLSTM2D, Conv3DTranspose, Concatenate, Dropout
from keras import backend as K
from keras.callbacks import History 
from keras import losses, optimizers
from keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
#from sklearn.metrics import mean_squared_error
path='Data'

'''
This is file is the antoencoder neural network's structure.There are 4 different types of matrices with 2 length and 2 w.
And there are dataset of ground test and flight test. So we have computed 8 neural networks, 
specifically on 4 different types of matrices and ground and flight test.
'''
#Neural network

# The "length" and "length2" are the length to be used in the convolutional LSTM blocks. And the data correspond all have the same length with the framework.
length = 100
length2 = 20

# There are in total 8 frameworks for the Airbus Commercial Aircraft Challenge. There are two lengths, 100 and 20, two options of w, 15 and 5, and two different dataset, data of ground test and flight test. The 1st flight and ground autoencoder are of length=100 and w=5, the 2nd flight and ground autoencoder are of length=100 and w=15. The 3rd flight and ground autoencoder are of length=20 and w=5, and the 4th flight and ground autoencoder are of length=20 and w=15. Among them all, the 4th flight and ground test works the best on detecting anomalies.
## Structure of 1st and 2nd flight and ground autoencoder, of length=100. The 1st is with w=5, and the 2nd is with w=15. ##  
input_img1 = Input( shape=( len(data.params_of_interest), len(data.params_of_interest), length , 1 ), name = "Input") # adapt this if using `channels_first` image data format

# The convolutional part of the autoencoder, we can see the input, conv1 until conv4. 
conv1 = Conv3D(filters=16, kernel_size=(3,3,1), strides=(1,1,1), activation='selu', padding='same', data_format='channels_last', name='conv1')(input_img1)
conv2 = Conv3D(filters=32, kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', data_format='channels_last', name='conv2')(conv1)
conv3 = Conv3D(filters=64, kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', data_format='channels_last', name='conv3')(conv2)
conv4 = Conv3D(filters=128,kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', data_format='channels_last', name='conv4')(conv3)

# The convolutional LSTM (ConvLSTM) operators. Added after each convolutional and dropout operators, to detect all time dependencies from matrices to matrices.
convlstm1 = ConvLSTM2D(filters=16, return_sequences=True, kernel_size=(3,3), strides=(1,1), activation='selu', padding='same', data_format='channels_last', name='convlstm1')(conv1)
convlstm2 = ConvLSTM2D(filters=32, return_sequences=True, kernel_size=(3,3), strides=(1,1), activation='selu', padding='same', data_format='channels_last', name='convlstm2')(conv2)
convlstm3 = ConvLSTM2D(filters=64, return_sequences=True, kernel_size=(3,3), strides=(1,1), activation='selu', padding='same', data_format='channels_last', name='convlstm3')(conv3)
convlstm4 = ConvLSTM2D(filters=128,return_sequences=True, kernel_size=(3,3), strides=(1,1), activation='selu', padding='same', data_format='channels_last', name='convlstm4')(conv4)

# The deconvolutional part of the autoencoder, to reconstrut the input matrices. 
deconv4 = Conv3DTranspose(filters=64 , kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', output_padding=(1,1,0), data_format='channels_last', name='deconv4')(convlstm4)
concat4 = Concatenate(axis=4, name='concat4')([convlstm3, deconv4])
deconv3 = Conv3DTranspose(filters=32 , kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', output_padding=(1,1,0), data_format='channels_last', name='deconv3')(concat4)
concat3 = Concatenate(axis=4, name='concat3')([convlstm2, deconv3])
deconv2 = Conv3DTranspose(filters=16 , kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', output_padding=(1,1,0), data_format='channels_last', name='deconv2')(concat3)
concat2 = Concatenate(axis=4, name='concat2')([convlstm1, deconv2])
deconv1 = Conv3DTranspose(filters=1  , kernel_size=(3,3,1), strides=(1,1,1), activation='selu', padding='same' , data_format='channels_last', name='deconv1')(concat2)

###########################################################################################################################
## Structure of 3rd and 4th flight and ground autoencoder, of length=20.  The 3rd is with w=5, and the 4th is with w=15. ##  
input_img2= Input( shape=( len(data.params_of_interest), len(data.params_of_interest), length2, 1 ), name = "Input")

conv12 = Conv3D(filters=16, kernel_size=(3,3,1), strides=(1,1,1), activation='selu', padding='same', data_format='channels_last', name='conv1')(input_img2)
conv22 = Conv3D(filters=32, kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', data_format='channels_last', name='conv2')(conv12)
conv32 = Conv3D(filters=64, kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', data_format='channels_last', name='conv3')(conv22)
conv42 = Conv3D(filters=128,kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', data_format='channels_last', name='conv4')(conv32)

convlstm12 = ConvLSTM2D(filters=16, return_sequences=True, kernel_size=(3,3), strides=(1,1), activation='selu', padding='same', data_format='channels_last', name='convlstm1')(conv12)
convlstm22 = ConvLSTM2D(filters=32, return_sequences=True, kernel_size=(3,3), strides=(1,1), activation='selu', padding='same', data_format='channels_last', name='convlstm2')(conv22)
convlstm32 = ConvLSTM2D(filters=64, return_sequences=True, kernel_size=(3,3), strides=(1,1), activation='selu', padding='same', data_format='channels_last', name='convlstm3')(conv32)
convlstm42 = ConvLSTM2D(filters=128,return_sequences=True, kernel_size=(3,3), strides=(1,1), activation='selu', padding='same', data_format='channels_last', name='convlstm4')(conv42)

deconv42 = Conv3DTranspose(filters=64 , kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', output_padding=(1,1,0), data_format='channels_last', name='deconv4')(convlstm42)
concat42 = Concatenate(axis=4, name='concat4')([convlstm32, deconv42])
deconv32 = Conv3DTranspose(filters=32 , kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', output_padding=(1,1,0), data_format='channels_last', name='deconv3')(concat42)
concat32 = Concatenate(axis=4, name='concat3')([convlstm22, deconv32])
deconv22 = Conv3DTranspose(filters=16 , kernel_size=(2,2,1), strides=(2,2,1), activation='selu', padding='same', output_padding=(1,1,0), data_format='channels_last', name='deconv2')(concat32)
concat22 = Concatenate(axis=4, name='concat2')([convlstm12, deconv22])
deconv12 = Conv3DTranspose(filters=1  , kernel_size=(3,3,1), strides=(1,1,1), activation='selu', padding='same' , data_format='channels_last', name='deconv1')(concat22)

##############
autoencoder_flight1 = Model(input_img1, deconv1)
autoencoder_flight1.compile(optimizer='adam', loss='mse', metrics=['mae'])
##############
autoencoder_flight2 = Model(input_img1, deconv1)
autoencoder_flight2.compile(optimizer='adam', loss='mse', metrics=['mae'])
##############
autoencoder_flight3 = Model(input_img2, deconv12)
autoencoder_flight3.compile(optimizer='adam', loss='mse', metrics=['mae'])
##############
autoencoder_flight4 = Model(input_img2, deconv12)
autoencoder_flight4.compile(optimizer='adam', loss='mse', metrics=['mae'])


############
autoencoder_ground1 = Model(input_img1, deconv1)
autoencoder_ground1.compile(optimizer='adam', loss='mse', metrics=['mae'])
#############
autoencoder_ground2 = Model(input_img1, deconv1)
autoencoder_ground2.compile(optimizer='adam', loss='mse', metrics=['mae'])
#############
autoencoder_ground3 = Model(input_img2, deconv12)
autoencoder_ground3.compile(optimizer='adam', loss='mse', metrics=['mae'])
#############
autoencoder_ground4 = Model(input_img2, deconv12)
autoencoder_ground4.compile(optimizer='adam', loss='mse', metrics=['mae'])
############

if __name__ == '__main__':
    # In the main executing program, we will only show you the framework structures of autoencoders for length=100 and length=20. 
    print("#######################")
    print("The structure of autoencoder of length 100: ")
    print("#######################")
    autoencoder_ground2.summary()
    print("#######################")
    print("The structure of autoencoder of length 20: ")
    print("#######################")
    autoencoder_ground4.summary()
