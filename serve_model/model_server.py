# load libraries
import flask
import io
import inspect
import keras
import os
import numpy as np
import tensorflow as tf
import time
import sys
from PIL import Image

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


# global variables
app = flask.Flask(__name__)
model = None
graph = None

def modify_path():
    """
    add parent directory to path
    """
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir) 

def setup_tf_session():
    """"
    configure tensorflow session
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(sess)

def load_model(model_path):
    """
    load retinanet model
    """
    
    global model
    model = models.load_model(model_path, backbone_name='resnet50')
    global graph
    graph = tf.get_default_graph()
    # convert model to inference model
    model = models.convert_model(model)

@app.route("/predict", methods=["GET","POST"])
def predict():
    """
    define a predict function as an endpoint
    """
    
    data = {"success": False}
    if flask.request.files.get("image"):
        # read image from request
        image = flask.request.files["image"].read()
        # convert image to BGR
        image = read_image_bgr(io.BytesIO(image))
        # preprocess image for model
        image = preprocess_image(image, mode='pass')
        image, scale = resize_image(image)
        data["scale"] = scale

        # process image
        with graph.as_default():
            start_time = time.time()
            # generate prediction bounding boxes, scores, and labels on the input image
            boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
            # add inference time to data dictionary
            data["time"] = time.time() - start_time

            # add prediction boxes, scores, & labels to data dictionary
            data["predictions"] = {"boxes": boxes.tolist(),
                                "scores": scores.tolist(),
                                "labels": labels.tolist()}

        # prediction was successful
        data["success"] = True
    
    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("** Loading defect detection model and starting server..."
            "please wait until server has fully started"))
    modify_path()
    # load the model
    path = '../keras_retinanet/bin/snapshots/resnet50_csv_20-MXT-100px.h5'
    load_model(path)
    # start the flask app, allow remote connections
    app.run()