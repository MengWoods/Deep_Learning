#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:46:17 2019

@author: menghao
"""

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os

import argparse

'''========================================
            hyper-parameters
   ========================================'''

'''path'''
root_path = os.getcwd() # get current working directory
#os.chdir('/Users/menghao/Document/2_project/1_HEUbuilding/testpics')

img_path = '/Users/menghao/Document/2_project/1_HEUbuilding/testpics/'
model_path = '/Users/menghao/Document/2_project/1_HEUbuilding/result/2019_8_30/model.model'
pickle_path = '/Users/menghao/Document/2_project/1_HEUbuilding/result/2019_8_30/mlb.pickle'

SCALE = (192, 108)
# construct the argument parse and parse the arguments
# load the image

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--name', required=True, help='type in the pic\'s name under testpics/ without extension')
args = vars(ap.parse_args())
 
# pre-process the image for classification
pic_name = args['name']
image = cv2.imread(img_path + pic_name + '.jpg')
output = imutils.resize(image, width=400)

image = cv2.resize(image, SCALE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")
model = load_model(model_path)
mlb = pickle.loads(open(pickle_path, "rb").read())
# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]
# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))
 
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(8000)





