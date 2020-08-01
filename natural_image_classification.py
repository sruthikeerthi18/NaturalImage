# -*- coding: utf-8 -*-
"""Natural Image Classification.py
"""

#Importing Packages
from os import listdir

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K


#Values required
img_width, img_height = 224, 224
batch_size = 32

epochs = 5
steps_per_epoch = 200
validation_steps = 20

#Setting the image shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

category = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
#classes => number
cat_to_class = {i : c for c, i in enumerate(category) }
class_to_cat = {c : i for c, i in enumerate(category) }

#data paths
train_data_path = 'natural_images/train'
test_data_path = 'natural_images/test'

#preprocess
def preprocess(data_path):
    images = []
    classes = []
    files = listdir(data_path)
    shuffle(files)
    for f in files:
        img = load_img(data_path + '/' + f, target_size= input_shape)
        img_array = img_to_array(img)
        c = f.split("_")[0]
        images.append(img_array)
        classes.append(cat_to_class[c])
    return np.array(images), np.array(classes)        


train_images, train_classes = preprocess(train_data_path)
test_images, test_classes = preprocess(test_data_path)

ohe = OneHotEncoder()
train_classes = ohe.fit_transform(train_classes.reshape(-1,1)).toarray()


"""Manual Sequence Model

Arch:  

CONV2D(3,3) * 16 RELU + MaxPool(2,2)   

CONV2D(3,3) * 32 RELU + MaxPool(2,2)

CONV2D(3,3) * 64 RELU + MaxPool(2,2)

CONV2D(3,3) * 64 RELU + MaxPool(2,2)

Flatten

Dense 256 RELU

Dense 64 RELU

Dropout 0.5

Dense 8 SOFTMAX
"""

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=  16, kernel_size = (3, 3), padding = "same", input_shape = input_shape))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding ="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding ="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Activation("relu"))

model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation("relu"))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Activation('softmax'))

model.summary()

#Using adam
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#Fitting the model to the train data generated
history=model.fit(x = train_images,
                  y = train_classes,
                  epochs = epochs,
                  batch_size = batch_size,
                  validation_split = 0.2)


#Plotting train and val accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Plotting train and val loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


predict_model1 = model.predict(test_images)

#test accuracy of model 1
predictions_model1 = np.argmax(predict_model1,axis=1)

from sklearn.metrics import accuracy_score

acc_model1 = accuracy_score(predictions_model1,test_classes)

#Savinf the trained model
model.save('model_1.h5')

"""Model 2 Mobile Net V2"""

model_mnv2 = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top = False,
                                                            pooling = 'avg',
                                                            weights = 'imagenet',
                                                            input_shape = input_shape)

model_2 = tf.keras.models.Sequential(layers=[model_mnv2,tf.keras.layers.Dropout(0.5),tf.keras.layers.Dense(8,activation='softmax')])
model_2.layers[0].trainable = False

model_2.summary()

#Using adam
model_2.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

history_2=model_2.fit(x = train_images,
                      y = train_classes,
                      epochs = 5,
                      batch_size = batch_size,
                      validation_split = 0.2)


#Plotting train and val accuracy
plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Plotting train and val loss
plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

predict_model2 = model_2.predict(test_images)

#test accuracy of model 1
predictions_model2 = np.argmax(predict_model2,axis=1)

acc_model2 = accuracy_score(predictions_model2,test_classes)

#Savinf the trained model
model_2.save('model_2.h5')

"""Model 3 VGG16"""

model_vgg16 = tf.keras.applications.vgg16.VGG16(include_top = False,
                                                pooling = 'avg',
                                                weights = 'imagenet',
                                                input_shape = input_shape)

model_3 = tf.keras.models.Sequential(layers=[model_vgg16,tf.keras.layers.Dropout(0.5),tf.keras.layers.Dense(8,activation='softmax')])
model_3.layers[0].trainable = False

model_3.summary()

#Using adam
model_3.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

history_3 = model_3.fit(x = train_images,
                        y = train_classes,
                        epochs = 5,
                        batch_size = batch_size,
                        validation_split = 0.2)


#Plotting train and val accuracy
plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Plotting train and val loss
plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

predict_model3 = model_3.predict(test_images)

#test accuracy of model 1
predictions_model3 = np.argmax(predict_model3,axis=1)

acc_model3 = accuracy_score(predictions_model3,test_classes)

#Savinf the trained model
model_3.save('model_3.h5')

