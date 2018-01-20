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
#skilearn for preprocessing
from sklearn import preprocessing
#it will shuffle the data to better learning of network
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

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
        print (img_data.shape)
    else:
        img_data=np.expand_dims(img_data_list,axis=4)
        print(img_data.shape)

else:
    if K.image_dim_ordering()=='th':
        img_data=np.rollaxis(img_data,3,1)
        print(img_data.shape)

#3

def image_to_feature_vector(image, size=(128,128)):
    #image will be resized to fixed size, then flatten the image into
    #a list of raw pixel intensities
    return cv2.resize(image,size).flatten()

img_data_list=[]
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+dataset)
    print('Loaded the images of dataset-'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path+'/'+dataset+'/'+img)
        input_img=cv2.cvtColor(input_img,cv2.COLOR_BAYER_BG2GRAY)
        input_imp_flatten=image_to_feature_vector(input_img,(128,128))
        img_data_list.append(input_imgP_flatten)

img_data=np.array(img_data_list)
img_data=img_data.astype('float32')

print(img_data.shape)

img_data_scaled=preprocessing.scale(img_data)
print(img_data_scaled.shape)

print(np.mean(img_data_scaled))
print(np.std(img_data_scaled))

print(img_data_scaled.mean(axis=0))#check
print(img_data_scaled.std(axis=0))

if K.image_dim_ordering()=='th':
    img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
    print (img_data_scaled.shape)
else:
    img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
    print(img_data_scaled.shape)

#4
#define number of classes
num_classes=4

num_of_samples=img_data.shape[0]
labels=np.ones((num_of_samples,),dtype='int64')

labels[0:102]=0
labels[102:204]=1
labels[204:606]=2
labels[606:]=3

names=['cats','dogs','horses','humans']

#converst class labels to on-hot encoding
Y=np_utils.to_categorical(labels,num_classes)

#5
#shuffle the data--> as different data(validation and testing data is not present)
x,y=shuffle(img_data,Y,random_state=2)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

#6
#train and build the Convolution Neural network
input_shape=img_data_scaled[0].shape

model=Sequential()

model.add(Convolution2D(32,3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adedelta',metrics=["accuracy"])
#Training
#hist=model.fit(X_train,y_train,batch_size=32,nb_epoch=20,verbose=1,validation_data=(X_test,y_test))
#or
hist=model.fit(X_train,y_train,batch_size=32,nb_epoch=20,verbose=1,validation_split=0.2)
#20 epoch is very we may need to increase it to find which epoch
#is better by using visualisation

#7
#Visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])

plt.style.use(['classic'])#find more

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.tittle('accuracy')
plt.grid(True)
plt.legend(['train','val'],loc=4)

plt.style.use(['classsic'])