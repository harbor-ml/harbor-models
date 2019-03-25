from clipper_admin import DockerContainerManager, ClipperConnection

clipper_conn = ClipperConnection(DockerContainerManager())

clipper_conn.connect()

clipper_conn.register_application(
				name="ssd-mobilenet-object-detection-tf",
				input_type="bytes",
				default_output="default",
				slo_micros=9000000) # 9s

clipper_conn.deploy_model(
	name="tf-ssd",
	version=0,
	input_type="bytes",
	image="tf-ssd",
	batch_size=1)

clipper_conn.link_model_to_app("ssd-mobilenet-object-detection-tf", "tf-ssd")

print("Model Now Live at:", clipper_conn.get_query_addr())
