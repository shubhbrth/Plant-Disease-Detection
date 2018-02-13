from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

#%%
#Constructing the argument parser and parse arguments

ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="path to trained model")
ap.add_argument("-i","--image",required=True,help="path to inpur image")
args=vars(ap.parse_args())

#%%
#Load and Pre-Process the image

image=cv2.imread(args["image"])
orig=image.copy()

image=cv2.resize(image,(128,128))
image=image.astype("float")/255.0
image=img_to_array(image,axis=0)

#%%
#Load Trained CNN
print("[INFO] loading network...")
model=load_model(args["model"])

#%%Building and Drawing the label on the image
##Label Build
#label = "Santa" if santa > notSanta else "Not Santa"
#proba = santa if santa > notSanta else notSanta
#label = "{}: {:.2f}%".format(label, proba * 100)

## Embed the label
#output = imutils.resize(orig, width=400)
#cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (0, 255, 0), 2)

## Output the Image
#cv2.imshow("Output", output)
#cv2.waitKey(0)

#%%
#how to use this model
#python test_network.py --model santa_not_santa.model \	--image examples/santa_01.png