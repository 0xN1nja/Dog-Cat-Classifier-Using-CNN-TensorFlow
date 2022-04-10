# Dog/Cat Classifier
A Model To Predict Dog/Cat. Got 83% Accuracy And 65% Validation Accuracy (Because Of Overfitting)
# Author
Abhimanyu Sharma, https://github.com/0xN1nja
# How To Use This Model
## Clone This Repository
```
git clone https://github.com/0xN1nja/Dog-Cat-Classifier-Using-CNN-TensorFlow.git
```
## Install Required Modules
```
pip install tensorflow numpy opencv-python
```
## Usage
```python
import tensorflow as tf
import cv2
import numpy as np
LABELS=["Dog","Cat"]
def prepare_data(filepath):
	img=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
	img=cv2.resize(img,(50,50))
	return img.reshape(-1,50,50,1)
img_path=r"" # Add Your Image's Path
model=tf.keras.models.load_model("64x3-CNN.h5")
# Predict
print(LABELS[int(model.predict([prepare_data(img_path)])[0])])
```