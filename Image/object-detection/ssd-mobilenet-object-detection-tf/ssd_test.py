import base64
import json
import requests

IMAGE_FILE = "../old_object_detection/detection_test.jpg"
ADDR = "localhost:1337"
MODEL_NAME = "ssd-mobilenet-object-detection-tf"

def query(addr, model_name, filename):
	url = "http://%s/%s/predict" % (addr, model_name)
	req_json = json.dumps({
		"input":
		base64.b64encode(open(filename, "rb").read()).decode() # bytes to unicode
	})
	headers = {'Content-type': 'application/json'}
	r = requests.post(url, headers=headers, data=req_json)
	print(r.json())

query(ADDR, MODEL_NAME, IMAGE_FILE)