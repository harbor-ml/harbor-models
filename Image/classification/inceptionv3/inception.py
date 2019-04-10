import tensorflow as tf
import numpy as np
import base64
import io
import os
import PIL
import tempfile

def encode_image(image_path):
	raw_bytes = open(image_path, "rb").read()
	encoded_bytes = base64.b64encode(raw_bytes)
	return encoded_bytes

def predict_tf_inception(sess, imgs):
	img_tensors = []
	for img in imgs:
		image = PIL.Image.open(io.BytesIO(img))
		image = image.resize((299, 299))
		image = tf.keras.preprocessing.image.img_to_array(image)
		image = tf.keras.applications.inception_v3.preprocess_input(image)
		img_tensors.append(image)

	preds = sess.run('predictions/Softmax:0', feed_dict={'img_input:0': img_tensors})
	labels = tf.keras.applications.inception_v3.decode_predictions(preds)
	return labels

def preprocess(img_path):
	img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
	img = tf.keras.preprocessing.image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = tf.keras.applications.inception_v3.preprocess_input(img)
	return img

# Setup Model
sess = tf.Session()
tf.keras.backend.set_session(sess)
input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='img_input')
# Input shape: (None, 224, 224, 3)
# Ouptut shape: (None, 1000)
model = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=input_tensor, input_shape=None, pooling=None, classes=1000)

# Testing Function
# img = preprocess(IMAGE_FILE)
# preds = sess.run('fc1000/Softmax:0', feed_dict={'img_input:0': img})
# label = tf.keras.applications.resnet50.decode_predictions(preds)
# print(label)

app_name = "inceptionv3-image-classification-keras"
default_output = "default"
model_name = "inceptionv3"

from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.tensorflow import deploy_tensorflow_model

clipper_conn = ClipperConnection(DockerContainerManager())

clipper_conn.start_clipper()

clipper_conn.register_application(name=app_name, input_type="bytes", default_output=default_output, slo_micros=40000000)

# Connect to an already-running Clipper cluster
# clipper_conn.connect()

deploy_tensorflow_model(
    clipper_conn,
    model_name,
    "1",
    "bytes",
    predict_tf_inception,
    sess,
    pkgs_to_install=['pillow'])

clipper_conn.link_model_to_app(app_name=app_name, model_name=model_name)

