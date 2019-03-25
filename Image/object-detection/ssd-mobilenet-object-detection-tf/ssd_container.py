import rpc
import os
import sys
import json

import numpy as np
import tensorflow as tf

import base64
import io
import os
import PIL
import tempfile
import pickle

### CONFIG ###
IMPORT_ERROR_RETURN_CODE = 3

MODEL_NAME = 'ssd_mobilenet'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "obj_detection_labels.p"

class TfContainer(rpc.ModelContainerBase):
	def __init__(self):


		self.graph = tf.Graph()
		with self.graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		self.sess = tf.Session(graph=self.graph)

		self.category_index = pickle.load(open(PATH_TO_LABELS, "rb" ))
		self.box_limit = 20
		self.min_threshold = 0.5

		self.tensor_dict = {}
		with self.graph.as_default():
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
			if 'detection_masks' in self.tensor_dict:
				# The following processing is only for single image
				detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
				# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				self.tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
			self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
		return

	def predict_bytes(self, inputs):
		img = inputs[0]
		tmp = tempfile.NamedTemporaryFile('wb', delete=False, suffix='.png')
		tmp.write(io.BytesIO(img).getvalue())
		tmp.close()
		image = PIL.Image.open(tmp.name)
		output_dict = self.sess.run(self.tensor_dict,
				feed_dict={self.image_tensor: np.expand_dims(image, 0)})
		# all outputs are float32 numpy arrays, so convert types as appropriate
		output_dict['num_detections'] = int(output_dict['num_detections'][0])
		output_dict['detection_classes'] = output_dict[
				'detection_classes'][0].astype(np.uint8)
		output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
		output_dict['detection_scores'] = output_dict['detection_scores'][0]
		if 'detection_masks' in output_dict:
			output_dict['detection_masks'] = output_dict['detection_masks'][0]
		im_width, im_height = image.size
		ret_list = []
		boxes = output_dict['detection_boxes']
		classes = output_dict['detection_classes']
		scores = output_dict['detection_scores']
		for i in range(min(self.box_limit, boxes.shape[0])):
			if scores is None or scores[i] > self.min_threshold:
				ymin, xmin, ymax, xmax = boxes[i].tolist()
				(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
									  ymin * im_height, ymax * im_height)
				category = self.category_index.get(classes[i])
				if category:
					desc = (category["id"], category["name"], scores[i], left, bottom, right, top)
					ret_list.append(str(desc))

		return [str(ret_list)]

if __name__ == "__main__":
	print("Starting Tf SSD Container")
	rpc_service = rpc.RPCService()
	try:
		model = TfContainer()
		print("Successfullyl instantiated model")
		sys.stdout.flush()
		sys.stderr.flush()
	except ImportError:
		sys.exit(IMPORT_ERROR_RETURN_CODE)
	rpc_service.start(model)