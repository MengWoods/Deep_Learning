#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:06:27 2019

@author: menghao
"""
import os
import cv2
import glob
from skimage import io, color
import numpy as np
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
#from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import random
import pickle


matplotlib.use("Agg")

'''========================================
            hyper-parameters
   ========================================'''
'''path'''
root_path = os.getcwd() # get current working directory
os.chdir('/Users/menghao/Document/2_project/1_HEUbuilding/1_code')
#path = os.path.join('../0_data/')
path = '../0_data/'
path1 = '../0_data/1/'
path2 = '../0_data/2/'
path3 = '/Users/menghao/Document/2_project/1_HEUbuilding/'+'2_prepaired_data/'
path4 = '../0_data/' # for saving npy file

'''training'''
EPOCHS = 75
INIT_LR = 1e-3
BS = 32

'''others'''
random.seed(42)
'''========================================
            functions
   ========================================'''
'''------reading date to array with labels------'''
def read_img(path):
    cate = [path+x for x in os.listdir(path) if os.path.isdir(path+x)] # list all sub directory
    #cate.pop(-1)
    imgs = []
    labels = []
    i, j, k, l, m = 0,0,0,0,0
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'): # glob.glob: open the directory, im is directory
            label = im[12:14]
            # A B C D 
            if label == 'A0':
                label = ['main_building_front',]
                name = 'main_building_front'
                picpath = path3 + name + '_' + str(i) + '.jpg'
                i+=1
            elif label == 'B0':
                label = ['main_building_back',]
                name = 'main_building_back'
                picpath = path3 + name + '_' + str(j) + '.jpg'
                j+=1
            elif label == 'C0':
                label = []
                name = ''
                picpath = path3 + name + '_' + str(k) + '.jpg'
                k+=1
            elif label == 'D0':
                label = ['dormitory']
                name = 'dormitory'
                picpath = path3 + name + '_' + str(l) + '.jpg'
                l+=1
            elif label == 'AD':
                label = ['main_building_front','dormitory']
                name = 'main_building_front__dormitory'
                picpath = path3 + name + '_' + str(m) + '.jpg'
                m+=1
            print('reading the images: %s; label: %s' % (im,label))
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # keep dims
#            cv2.imwrite(path3 + name + '_' + str(i) + '.jpg', img)
            img = img_to_array(img)
            imgs.append(img)
            labels.append(label)
    print ('[INFO] finish loading...')
    print ('[INFO] image dim:', img.shape)
    print ('[INFO] summary: main_building_front: %d, \
           main_building_back: %d, \
           dormitory: %d, \
           main_building_front__dormitory: %d,\
           other: %d' % (i,j,l,m,k))
    print ('[INFO] images saved on path:',path3)
    dim = img.shape
    dims = [dim[0], dim[1], dim[2]]
    return np.asarray(imgs, np.float32), labels, dims

'''========================================
            classes
   ========================================'''
'''---------Neural Network building---------'''
# NN according to: SmallVGG, https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
# import the necessary packages
class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
            
        # CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))  

		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
 
		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
 
		# use a *softmax* activation for single-label classification
		# and *sigmoid* activation for multi-label classification
		model.add(Dense(classes))
		model.add(Activation(finalAct))
 
		# return the constructed network architecture
		return model
'''========================================
            train
   ========================================'''
'''----------------arguments--------------------'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help = 'path to input dataset')
ap.add_argument('-m', '--model', required=True,
                help='path to output model')
ap.add_argument('-l', '--labelbin', required=True,
                help='path to output label binarizer')
ap.add_argument('-p', '--plot', type=str, default='plot.png',
                help='path to output accuracy/loss plot')
args = vars(ap.parse_args())
'''========================================
            train
   ========================================'''
def main():
    # data loading
    data, label, dims = read_img(path)
    np.save('HEU_data.npy', data)
    np.save('HEU_label.npy', label)
    
    # np.load('filename.npy')
    label = np.load('HEU_label.npy',allow_pickle = True)
    data = np.load('HEU_data.npy')
    
    HEIGHT, WIDTH, DEPTH = dims[0], dims[1], dims[2]
    #io.imshow(data[3,:,:])
    IMAGE_DIMS = (HEIGHT,WIDTH,DEPTH)
    # initialize the optimizer
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)   
    # binarize the labels using scikit-learn's special multi-label
    # binarizer implementation
    print('[INFO] class labels:')
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(label)
    # [0,0,0] represents [dormitory, mainbuilding_back, mainbuilding_front]
    # loop over each of the possible class labels and show them
    for (i, label) in enumerate(mlb.classes_):
        print('{}. {}'.format(i+1, label))
        
    # initialize the model using a sigmoid activation as the final layer
    # in the network so we can perform multi-label classification
    print("[INFO] compiling model...")
#    SmallerVGGNet = SmallerVGGNet()
    model = SmallerVGGNet.build(
    	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")
    
    '''----------dataset preprocessing-----------'''
    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
    	labels, test_size=0.2, random_state=42)
    
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    	horizontal_flip=True, fill_mode="nearest")
    
    '''--------------training-------------------'''
    # compile the model using binary cross-entropy rather than
    # categorical cross-entropy -- this may seem counterintuitive for
    # multi-label classification, but keep in mind that the goal here
    # is to treat each output label as an independent Bernoulli
    # distribution
    model.compile(loss="binary_crossentropy", optimizer=opt,
    	metrics=["accuracy"])      
    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(
    	aug.flow(trainX, trainY, batch_size=BS),
    	validation_data=(testX, testY),
    	steps_per_epoch=len(trainX) // BS,
    	epochs=EPOCHS, verbose=1)
    
    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
     
    # save the multi-label binarizer to disk
    print("[INFO] serializing label binarizer...")
    f = open(args["labelbin"], "wb")
    f.write(pickle.dumps(mlb))
    f.close()
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(args["plot"])
    
if __name__ == '__main__':
    main()










    
