from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D		
from keras.models import Sequential

import numpy as np


from keras.datasets import mnist

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.pooling import MaxPooling2D

from keras.utils import np_utils

from keras.optimizers import SGD

from keras.regularizers import l1_l2, l2


import gzip
import sys

import _pickle as cPickle
import pylab

from keras import backend as K

import matplotlib.pyplot as plt

import h5py


def plot_results(xlabel, ylabel, distribution, val_distribution):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(distribution)
    plt.plot(val_distribution)
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()


def raster_plot(matrix):

    print(matrix)
    plt.imshow(matrix,aspect='auto',cmap='gray_r')
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Cell', fontsize=18)
    plt.title('Raster plot of retinal ganglion cell responses \n to white noise', fontsize=20)
    plt.colorbar()
    plt.show()

def plot_input(matrix):
    plt.imshow(matrix,cmap='gray_r')
    plt.title('Example of white noise', fontsize=20)
    plt.show()

def onecell_plot(time, response):

    plt.plot(time,response)
    plt.title('Example of the response of one cell', fontsize=20)
    plt.show()


def simple_CNN(nb_filters,nb_conv,nb_pool,nb_classes, reg_val1, reg_val2):

    #~ a linear stack of layers
    model = Sequential()
		
    # note: the very first layer **must** always specify the input_shape
    model.add(Conv2D(nb_filters, (nb_conv,nb_conv),strides = 2,padding='valid',input_shape= (50,50,1),data_format='channels_last'))
    #,bias_initializer='zeros',
    
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool)))
    
    model.add(Flatten())
    #model.add(Dense(nb_classes,init='normal', W_regularizer=l1_l2(0., 1e-3),activity_regularizer=l1_l2(1e-3, 0.)))

    model.add(Dense(nb_classes,kernel_regularizer=l1_l2(reg_val1, reg_val2)))
    model.add(Activation('softmax'))
    #model.add(Activation('softplus'))
    #model.add(ParametricSoftplus())
    
    model.compile(loss='poisson',optimizer='adam',metrics=['accuracy'])

    return model

def my_softmax(array):

    #print('HOLA')
    array = array.astype('float')
    mx = np.max(array)

    #print(array-mx)
    #sys.exit()

    #print(array-mx)
    
    
    numerator = np.exp(array-mx)
    #print(numerator)

    #sys.exit()

    denominator = np.sum(numerator)
    #print(denominator)
    #print(numerator/denominator)
    return numerator/denominator
    

		
if __name__ == '__main__':
    print('HOLA')
    verbose = False
    plot = False

    test1 = True # Training and validation with the exact same data
    test2 = False # Training with a splited data accoding to validation_split input.

    test_set_size = 500
    sample_rate = 100.
    
    with h5py.File('/Users/frangaray/Desktop/DLab/Neuromorphic/Retina_example/deep-retina-tutorial-master/data/whitenoise.h5', 'r') as h:


            ###############################################################
            # FILE STRUCTURE:
            # Name: test
            # Subcategories:
            # 1. Repeats: It has the numbers of cells
            # 2. Response: It contains the firing rate for every 10ms
            # 3. Stimulus (Input): they are the images shown (50X50 pixels)
            # 4. Time: total time (approx. 5 [min] = 30011/[100 Hz* 60 s])
            ###############################################################
            
   
            ########### Train set #####################
            #train set for ? minutes
            ###########################################
   
            # as  first approach we are taking a small part of the data to train, and we will use the same set to test (to check model parameters) we should obtain the same precision for both sets.
   
   
            whitenoise_stimulus = np.array(h['test/stimulus'][:-test_set_size])
            whitenoise_response = np.stack([np.array(h['test/repeats/%s' %key][:,:-test_set_size]) for key in sorted(h['test/repeats'].keys())])
            whitenoise_psth = np.array(h['test/response/firing_rate_10ms'][:,:-test_set_size])
            time = np.array(h['test/time'][:-test_set_size])
		
            ############### Visualization of the data ######################
            
            if verbose:
                    print('Stimulus shape: ',whitenoise_stimulus.shape) 
                    print('Response_shape: ',whitenoise_response.shape)
                    print('firing_rate shape: ', whitenoise_psth.shape)
                    print('Time shape: ', time.shape)
                    #sys.exit()

            # Some plots
            if plot:
                    plot_input(whitenoise_stimulus[23995,0:50,0:50])
                    # Raster plot example
                    raster_plot(whitenoise_response[:,0,:])
                    # Plot for one cell
                    onecell_plot(time, whitenoise_response[16,0,:])

		



            ################## Test set ####################################
            #test set for ? minutes

            test_whitenoise_stimulus = np.array(h['test/stimulus'][-test_set_size:])
            test_whitenoise_response = np.stack([np.array(h['test/repeats/%s' %key][:,-test_set_size:]) for key in sorted(h['test/repeats'].keys())])
            test_whitenoise_psth = np.array(h['test/response/firing_rate_10ms'][:,-test_set_size:])
            test_time = np.array(h['test/time'][-test_set_size:])
		

            # Preparing the data format input_shape = (rows, col, channels) or data_format = 'channels_first' input_shape = (channels, rows, col)
            np.random.seed(1337) #For reproducibility

            shape_ord = (50,50,1)
            whitenoise_stimulus = whitenoise_stimulus.reshape((whitenoise_stimulus.shape[0],)+shape_ord)
            test_whitenoise_stimulus = test_whitenoise_stimulus.reshape((test_whitenoise_stimulus.shape[0],)+shape_ord)
        
            # Taking only one of the repeats....
            whitenoise_response_2 = whitenoise_response[:,0,:]
            whitenoise_response_2 = np.transpose(whitenoise_response_2)
            
            test_whitenoise_response_2 = test_whitenoise_response[:,0,:]
		
            
            # Taking 2000 samples randomly
            rand_samp = np.random.randint(1,29000,2000)
            whitenoise_stimulus = whitenoise_stimulus[rand_samp,::,::]
            whitenoise_response_2 = whitenoise_response_2[rand_samp,::]

            if verbose:
                    print('Training sample dimensions:')
                    print(whitenoise_stimulus.shape)
                    print(whitenoise_response_2.shape)


            # Test softmax

            #test_whitenoise_response_2[::,::]
            #print(test_whitenoise_response_2[::,::].shape)
            new_test = my_softmax(test_whitenoise_response_2[::,::])
            #print(new_test.shape)
            #raster_plot(new_test)
            #sys.exit()
            
            ######################### Simple CNN #################################################
            # -- Initializing the values for the convolution neural network
            
            
            nb_epoch = 200  # keep very low! Please increase if you have GPU
            batch_size = 1000
            nb_filters = 10 # number of convolutional filters to use
            nb_conv = 15 # convolution kernel size, filter size
            nb_pool = 10 # size of pooling area for max pooling
            nb_classes = 28 

            my_model = simple_CNN(nb_filters,nb_conv,nb_pool,nb_classes, 0., 0.)
            
            my_model.summary()
            

            if verbose:
                    print(my_model.get_config())
                    print(model.layers[0].get_weights())
                    print(model.layers[1].get_weights())
            #############################################################################################
            
            # Training
            if test1:
                hist = my_model.fit(whitenoise_stimulus, whitenoise_response_2, batch_size=batch_size, epochs=nb_epoch, verbose=2, validation_data=(whitenoise_stimulus, whitenoise_response_2))

            if test2:
                    hist = my_model.fit(whitenoise_stimulus, whitenoise_response_2, batch_size=batch_size, epochs=nb_epoch, verbose=2, validation_split = 0.2)

            # Model results:
            plot_results('Epochs', 'Loss',hist.history['loss'],hist.history['val_loss'])
            plot_results('Epochs', 'Accuracy',hist.history['acc'],hist.history['val_acc'])
            
		
            # Evaluating the model on the test data    
            loss, accuracy = my_model.evaluate(whitenoise_stimulus, whitenoise_response_2, verbose=0)
            print('Test Loss:', loss)
            print('Test Accuracy:', accuracy)


            # Looking at the prediction of the model with the test data.
            predictions = my_model.predict(test_whitenoise_stimulus, verbose = 1)

            # Raster plot with the predictions
            raster_plot(np.transpose(predictions[::,::]))
            # For comparison the raster plot with the true data.
            raster_plot(new_test)





			
