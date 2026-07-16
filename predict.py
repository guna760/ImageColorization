import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("models/colorization_model.keras")

img = cv2.imread("dataset/test/test.jpg")

img = cv2.resize(img,(128,128))

lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

L = lab[:,:,0]/255.0

input_img = L.reshape(1,128,128,1)

pred = model.predict(input_img)

pred = pred[0]*128

lab[:,:,1:] = pred.astype(np.uint8)

color = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

cv2.imwrite("outputs/result.jpg",color)

print("Color Image Saved")