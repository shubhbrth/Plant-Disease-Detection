import os,cv2
import numpy as np

PATH = os.getcwd()
# Define data path
data_path = PATH + '\path_test'
data_path.replace('\\', '/')

img=cv2.imread(data_path+'/'+'dimg6.jpg')
img=cv2.resize(img,(64,64))
img=img.astype('float32')
img/=255
img.shape
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img=np.expand_dims(img,axis=1)
img=np.rollaxis(img,2,1)
img=np.expand_dims(img,axis=0)
img=np.rollaxis(img,3,1)

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

result=loaded_model.predict(img)
max=result[0][0]

for i in range(0,len(result[0])):
	if result[0][i]>max:
		max=result[0][i]
		label=i
		
PATH = os.getcwd()
# Define data path
data_path = PATH + '\Dataset'
data_path.replace('\\', '/')
data_dir_list = os.listdir(data_path)

category=data_dir_list
print(max)
print('last value:',label)
print('Model Predicted this as',category[label])
3