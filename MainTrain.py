import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D , Activation , Flatten , Dense,Dropout
from keras.utils import to_categorical


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_dir = os.path.join(BASE_DIR, 'datasets')
no_dir = os.path.join(dataset_dir, 'no')
yes_dir = os.path.join(dataset_dir, 'yes')

dataset = []
label = []

INPUT_SIZE=64;

# NO TUMOUR (label = 0)
for image_name in os.listdir(no_dir):
    if image_name.lower().endswith('.jpg'):
        img_path = os.path.join(no_dir, image_name)

        image = cv2.imread(img_path)
        if image is None:
            print("Failed to load:", img_path)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize(( INPUT_SIZE,INPUT_SIZE))

        dataset.append(np.array(image))
        label.append(0)

# YES TUMOUR (label = 1)
for image_name in os.listdir(yes_dir):
    if image_name.lower().endswith('.jpg'):
        img_path = os.path.join(yes_dir, image_name)

        image = cv2.imread(img_path)
        if image is None:
            print("Failed to load:", img_path)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((INPUT_SIZE,INPUT_SIZE))

        dataset.append(np.array(image))
        label.append(1)

print("Total images:", len(dataset))
print("Total labels:", len(label))

# print(dataset)
# print(label)

dataset =np.array(dataset)
label =np.array(label)

x_train , x_test , y_train , y_test = train_test_split(dataset , label , test_size = 0.2 , random_state = 42)

print("x_train shape:" , x_train.shape) 


# binary classification problem
x_train = x_train / 255.0
x_test = x_test / 255.0

# x_test = normalize(x_test , axis = 1)
y_train = to_categorical(y_train , num_classes=2)
y_test = to_categorical(y_test , num_classes = 2)

#model buidling

model = Sequential()
model.add(Conv2D(32 , (3,3) , input_shape =(INPUT_SIZE , INPUT_SIZE , 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32 , (3,3) , kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32 , (3,3) , kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# model.add(Activation('sigmoid'))

# Binary CrossEntropy = 1 , sigmoid
#  Categorical Cross Entryopy = 2 , softmax


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics =['accuracy'] )

model.fit(x_train , y_train ,
          batch_size= 20 ,
          verbose = 1,
          epochs=10 , 
          validation_data = (x_test , y_test),
          shuffle=True 
)

model.save('BrainTumor10EpochsCategorical.h5')



