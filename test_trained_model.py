from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2,os

#%%
#Constructing the argument parser and parse arguments

#ap=argparse.ArgumentParser()
#ap.add_argument("-m","--model",required=True,help="path to trained model")
#ap.add_argument("-i","--image",required=True,help="path to inpur image")
#args=vars(ap.parse_args())
#%%
img_rows=64
img_cols=64
PATH = os.getcwd()
# Define data path
data_path = PATH + '\path_test'
data_path.replace('\\', '/')



#%%
#Load and Pre-Process the image
img_name='dimg4.jpg'
image=cv2.imread(data_path+'/'+img_name)
orig=image.copy()

image=cv2.resize(image,(64,64))
image=image.astype('float32')
image/=255
image.shape
image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image=np.expand_dims(image,axis=1)
image=np.rollaxis(image,2,1)
image=np.expand_dims(image,axis=0)
image=np.rollaxis(image,3,1)

#%%
#Load Trained CNN
print("[INFO] loading network...")
#model=load_model(args["model"])
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#%%Building and Drawing the label on the image
#Label Build
result=loaded_model.predict(image)
max=result[0][0]
label=0
for i in range(0,len(result[0])):
	if result[0][i]>max:
		max=result[0][i]
		label=i
category=['Diseased Cotton Plant','Healthy Cotton Plant']
out=category[label]
label = "{}: {:.2f}%".format(out, max * 100)

# Embed the label
output=cv2.resize(orig,(512,512))
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# Output the Image
cv2.imshow("Output", output)
cv2.imwrite(r'C:\Users\shubh\Documents\python workspace\Check\path_test\result'+'/'+img_name, output)
#%%
#how to use this model
#python test_network.py --model santa_not_santa.model \	--image examples/santa_01.png