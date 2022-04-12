# Dog/Cat Classifier
A Model To Predict Dog/Cat. Achieved 95.62% Accuracy And 79.73% Validation Accuracy (Because Of Overfitting)
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
# Usage
## On Image
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
model=tf.keras.models.load_model(r"64x3-CNN.h5")
# Predict
print(LABELS[int(model.predict([prepare_data(img_path)])[0])])
```
## Real-Time Prediction
```python
import tensorflow as tf
import cv2
import numpy as np
cap=cv2.VideoCapture(0) # Add Your Webcam Index
model=tf.keras.models.load_model(r"64x3-CNN.h5")
while True:
    _,img=cap.read()
    pred_img=cv2.resize(img,(50,50))
    pred_img=pred_img.reshape(-1,50,50,1)
    # Predict
    label=LABELS[int(model.predict([pred_img])[0])]
    cv2.putText(img,str(label),(70,70),color=(255,0,0),fontScale=2,fontFace=cv2.FONT_HERSHEY_COMPLEX)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
```
## Outputs
![Dog Prediction](https://media.discordapp.net/attachments/959703182718672946/963442547965976686/unknown.png?width=804&height=670)
![Cat Prediction](https://media.discordapp.net/attachments/959703182718672946/963442558447534110/unknown.png?width=782&height=670)
## Contributing
Pull Requests Are Welcome. For Major Changes, Please Open An Issue First To Discuss What You Would Like To Change.

Please Make Sure To Update Tests As Appropriate.