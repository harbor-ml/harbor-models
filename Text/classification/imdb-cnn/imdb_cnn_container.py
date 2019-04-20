import rpc
import os
import sys
import json
import keras

### CONFIG ###
IMPORT_ERROR_RETURN_CODE = 3

MODEL_NAME = 'imdb-cnn.hdf5'


class TfContainer(rpc.ModelContainerBase):
	def __init__(self):
		self.model = keras.models.load_model(MODEL_NAME)
		self.word_index = keras.datasets.imdb.get_word_index()
		return

	def predict_strings(self, inputs):
		vecs = []
		for text in inputs:
			vecs.append([1] + [word_index.get(word, 2) for word in text.split(' ')])
		seqs = keras.preprocessing.sequence.pad_sequences(vecs, maxlen=maxlen)
		o = sess.graph.get_tensor_by_name('activation_1/Sigmoid:0')
		i = sess.graph.get_tensor_by_name('input_1:0')
		preds = sess.run(o, feed_dict={i, seqs})
		for i in range(len(preds)):
			if preds[i] > 0.5:
				preds[i] = 'Positive'
			else:
				preds[i] = 'Negative'
		return preds

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