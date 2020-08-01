# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:42:59 2020

@author: Sruthi Keerthi
"""

import tensorflow as tf
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

from flask import Flask, request, render_template
#from gevent.pywsgi import WSGIServer

category = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
cat_to_class = {i : c for c, i in enumerate(category) }
class_to_cat = {c : i for c, i in enumerate(category) }
img_width, img_height = 224, 224

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def preprocess_input(path):
    img=load_img(path, target_size= input_shape)
    img_array=img_to_array(img)
    return img_array

def model_predict(img_path, model):
    img_input = preprocess_input(img_path)
    preds = model.predict(img_input)
    return preds


model=tf.keras.models.load_model('model_final.h5')
model.summary()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        base_path = 'natural_images/test'
        file_path = base_path + '/'+ f.filename
        preds = model_predict(file_path, model)

        pred_class = np.argmax(preds)
        result = class_to_cat[pred_class]
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
