from clipper_admin import DockerContainerManager, ClipperConnection

clipper_conn = ClipperConnection(DockerContainerManager())

clipper_conn.connect()

clipper_conn.register_application(
				name="imdb-cnn-keras",
				input_type="strings",
				default_output="strings",
				slo_micros=5000000) # 5s

clipper_conn.deploy_model(
	name="sentiment-cnn",
	version=0,
	input_type="strings",
	image="imdb-cnn-keras")

clipper_conn.link_model_to_app("imdb-cnn-keras", "sentiment-cnn")

print("Model Now Live at:", clipper_conn.get_query_addr())
