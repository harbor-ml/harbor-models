import rpc
import os
import sys
import json
import tensorflow as tf

### CONFIG ###
IMPORT_ERROR_RETURN_CODE = 3

MODEL_NAME = 'imdb-cnn.hdf5'


class TfContainer(rpc.ModelContainerBase):
	def __init__(self):
		self.model = tf.keras.models.load_model(MODEL_NAME)
		self.word_index = tf.keras.datasets.imdb.get_word_index()
		self.maxlen = 400
		return

	def predict_strings(self, inputs):
		inputs = [str(bytes, 'utf-8') for bytes in inputs]
		vecs = []
		for text in inputs:
			vecs.append([1] + [self.word_index.get(word, 2) for word in text.split(' ')])
		seqs = tf.keras.preprocessing.sequence.pad_sequences(vecs, maxlen=self.maxlen)
		preds = self.model.predict(seqs)
		results = []
		for i in range(len(preds)):
			if preds[i] > 0.5:
				results.append('Positive')
			else:
				results.append('Negative')
		return results

if __name__ == "__main__":
	print("Starting Keras Sentiment CNN Container")
	rpc_service = rpc.RPCService()
	try:
		model = TfContainer()
		print("Successfullyl instantiated model")
		sys.stdout.flush()
		sys.stderr.flush()
	except ImportError:
		sys.exit(IMPORT_ERROR_RETURN_CODE)
	rpc_service.start(model)
