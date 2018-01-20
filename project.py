# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 22:39:29 2018

@author: shubham
"""
import os,cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

#1
img_rows=128
img_cols=128
num_channel=1 #change to 3 if 3rgb is required

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+dataset)
    print('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path+'/'+dataset+'/'+img)
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BAYER_BG2GRAY)
        input_img_resize=cv2.resize(input_img,(128,128))
        img_data_list.append(input_img_resize)

img_data=np.array(img_data_list)
img_data=img_data.astype('float32')
img_data/=255
print(img_data.shape)

#2
if num_channel==1:
    if K.image_dim_ordering()=='th':
        img_data=np.expand_dims(img_data,axis=1)
        print (img)