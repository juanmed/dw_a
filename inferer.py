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

from difflib import SequenceMatcher
import easyocr
import json

images_path = "./images"
model_path = "./model"

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

def get_image_crop(image, xmin_n, ymin_n, xmax_n, ymax_n, margin_x=0, margin_y=0):
  h,w,d = image.shape
  xmin = int(xmin_n * w)
  ymin = int(ymin_n * h)
  xmax = int(xmax_n * w)
  ymax = int(ymax_n * h)
  return image[ymin - margin_y: ymax + margin_y, xmin - margin_x: xmax + margin_x]

def compare_strings(a,b):
  return SequenceMatcher(None, a, b).ratio()

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
        self.drugs_list = ['그린노즈에스','래피노즈','래피콜노즈','레피콜','베아제','소하자임','속시판','속코','시노카엔','시노타딘','시로제노','쎄르텍','씨콜드','알러샷','알러엔','알러지성 비염 코감기','알지싹로라','오로친','우라사','이지엔6이브','이지엔6애니','이지엔6에이스','이지엔6프로','지르텍','코드랍','코린투에스','코메키나','코미','코스펜','코졸텍','큐자임','클라리틴','프리노즈']
        self.threshold = 0.5

    def preprocess(self, image): 
        #image = tf.image.resize(image, (self.image_size, self.image_size)) 
        return tf.cast(image, tf.uint8) #/ 255.0 

    def infer(self, image=None): 
        print(" *** INICIANDO PROCESO ***")
        image_np_with_detections = image.copy()
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
        

        dets = {}
        for i,(box,cl,score) in enumerate(zip(detections['detection_boxes'],detections['detection_classes'],detections['detection_scores'])):
            if score > 0.3:
                if cl == 5: #is title
                    title_crop = get_image_crop(image_np_with_detections,box[1],box[0],box[3],box[2], 16, 16)
                    print("TITLE CROP: ",title_crop.shape)

                    #gtlabel = gt_labels[gt_labels['files']==image_path]['names'].tolist()[0]

                    if title_crop.shape[0] > title_crop.shape[1]:
                        title_crop=cv2.rotate(title_crop, cv2.ROTATE_90_CLOCKWISE)            

                    dim = (380,160)
                    title_crop = cv2.resize(title_crop, dim, interpolation = cv2.INTER_AREA)

                    for i in range(2):
                        reader = easyocr.Reader(['ko','en'], gpu=True) # need to run only once to load model into memory
                        result = reader.readtext(title_crop)
                        for detection in result:
                            points, text, score = detection
                            for drug in self.drugs_list:
                                score = compare_strings(drug,text)
                                if score >= self.threshold:
                                    try:
                                        dets[text].append(drug)
                                    except:
                                        dets[text] = [drug]
                        title_crop=cv2.flip(title_crop, -1)

        print("Result")
        print(dets)

        return dets#json.dumps(dets, ensure_ascii=False).encode('utf8')

if __name__ == '__main__':

    img = cv2.imread("./dll_package_recognition/pkrecog_031_0006.jpg")
    predictor = Inferer()
    output = predictor.infer(img)
    print(output)