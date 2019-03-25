import os
import subprocess
import base64
import io
import os
def yolo_pred(imgs):
    import base64
    import io
    import os
    import tempfile
    import subprocess

    num_imgs = len(imgs)
    ret_coords = []
    predict_procs = []
    file_names = []

    # First, we save the images to file
    for i in range(num_imgs):
        # Create a temp file to write to
        tmp = tempfile.NamedTemporaryFile('wb', delete=False, suffix='.jpg')
        tmp.write(io.BytesIO(imgs[i]).getvalue())
        tmp.close()
        file_names.append(tmp.name)

    # Second, we call ./darknet executable to detect objects in images.
    #   This is done in parallel.
    for file_name in file_names:
        process = subprocess.Popen(
            ['./darknet',
             'detector',
             'test',
             './cfg/coco.data',
             './cfg/yolov3-tiny.cfg',
             './yolov3-tiny.weights',
             file_name,
             '-json',
             '-dont_show',
             '-ext_output', '>',
             '{}.txt'.format(file_name+'_result')], stdout=subprocess.PIPE)
        predict_procs.append(process)

    # Lastly, we wait for all process to finished and return stdout of each process
    for process in predict_procs:
        process.wait()
        ret_coords += [' '.join(map(lambda byte_str: byte_str.decode(), process.stdout))]

    return ret_coords


# Do not be concerned if this cell takes a couple of seconds to run.
from clipper_admin import DockerContainerManager, ClipperConnection
clipper_conn = ClipperConnection(DockerContainerManager())

clipper_conn.connect()

from clipper_admin.deployers import python as python_deployer
python_deployer.deploy_python_closure(
    clipper_conn,
    name="yolov3",  # The name of the model in Clipper
    version=1,  # A unique identifier to assign to this model.
    input_type="bytes",  # The type of data the model function expects as input
    func=yolo_pred, # The model function to deploy
    base_image='clipper/darknet-yolov3-container'
)

clipper_conn.register_application(
    name="yolov3-object-detection-c",
    input_type="bytes",
    default_output="Default",
    slo_micros=10000000 # 10 seconds
)

clipper_conn.link_model_to_app("yolov3-object-detection-c", "yolov3")

print("Model Now Live at:", clipper_conn.get_query_addr())
