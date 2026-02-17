import cv2
from PIL import Image
from keras.models import load_model;
import numpy as np;

# model = load_model('BrainTumor10Epochs.h5')

model = load_model('BrainTumor10EpochsCategorical.h5')


image = cv2.imread('D:\\Projects\\Brain Tumour Detection\\pred\\pred0.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (64, 64))
img = image.astype('float32') / 255.0

# img = img / 255.0 
# print(img)

input_img = np.expand_dims(img ,axis = 0)

# result =model.predict_classes(input_img)


prediction = model.predict(input_img)
result = np.argmax(prediction, axis=1)

print(result)