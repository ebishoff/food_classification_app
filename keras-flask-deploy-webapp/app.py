from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pickle

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json
from keras.optimizers import Adam

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/nas_net_3_weights.h5'
JSON_PATH='models/nas_net_3.json'

# Load your trained model
# model = load_model(MODEL_PATH)
json_file=open(JSON_PATH,'r')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)
#load weights into new model
model.load_weights(MODEL_PATH)
#model.compile(optimizer=Adam(lr=.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(331, 331,3))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax(axis=-1) # Simple argmax
        pred_class_value=pred_class[0]
        
        #load label dictionary
        pkl_file=open('models/label_dictionary.pkl','rb')
        decoder=pickle.load(pkl_file)
        pkl_file.close()
        
        #load calorie dictionary
        pkl_file_2=open('models/calorie_dictionary.pkl','rb')
        calorie_decoder=pickle.load(pkl_file_2)
        pkl_file_2.close()
        
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = decoder[pred_class_value] + "\n" + calorie_decoder[pred_class_value]          # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
