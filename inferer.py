import pathlib as pl
import os
import gdown
import pathlib
import tensorflow as tf
import sys
import wget
import time
import tensorflow as tf 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile
import numpy as np

images_path = "./images"
model_path = "../model"

if not os.path.exists(images_path):
  pl.Path(images_path).mkdir(parents=True, exist_ok=True)
if not os.path.exists(model_path):
  pl.Path(model_path).mkdir(parents=True, exist_ok=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs: ".format(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Download and extract model

# old, all annotations https://drive.google.com/file/d/1r3dpmqPX-Vm3bCkhoAqPOKp6574_RU9p/view?usp=sharing
# final from Amazon Training https://drive.google.com/file/d/11QthlLdkC1u_bfzrECrKyOWSZK3gebL3/view?usp=sharing"
#url = 'https://drive.google.com/uc?id=11QthlLdkC1u_bfzrECrKyOWSZK3gebL3'
#output = 'ssd_540x640_batch8.zip'
#gdown.download(url, output, quiet=False)


# Create a ZipFile Object and load sample.zip in it
#with zipfile.ZipFile(output, 'r') as zipObj:
#   zipObj.extractall(model_path)

#labelmap_url = "https://raw.githubusercontent.com/juanmed/dw_a/main/dll_package_recognition/label_map.pbtxt"
#labelmap_file = wget.download(labelmap_url)

class Inferer:

    def __init__(self):

        PATH_TO_SAVED_MODEL = model_path + "/my_model" + "/saved_model"
        print('Loading model...', end='')
        start_time = time.time()
        # Load saved model and build the detection function
        self.model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))
        #self.predict = self.model.signatures["serving_default"]
        self.image_size = 3024

    def preprocess(self, image): 
        image = tf.image.resize(image, (self.image_size, self.image_size)) 
        return tf.cast(image, tf.uint8) #/ 255.0 

    def infer(self, image=None): 
        print(" *** INICIANDO PROCESO ***")
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32) 
        tensor_image = self.preprocess(tensor_image) 
        shape = tensor_image.shape 
        tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]]) 
        print(" *** PROCESO TERMINADO **** ")
        detections = self.model(tensor_image)#["detection_boxes"].numpy().tolist()#['conv2d_transpose_4']

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        detections['num_detections'] = num_detections 
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image.copy()

        return detections

if __name__ == '__main__':

    img = cv2.imread("./pkrecog_001_0001.jpg")
    predictor = Inferer()
    output = predictor.infer(img)
    print(output)