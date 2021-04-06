
import requests 
from PIL import Image 
import numpy as np 
import cv2
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import pathlib as pl
import os
import matplotlib.pyplot as plt

category_index = label_map_util.create_category_index_from_labelmap('label_map.pbtxt',
                                                                    use_display_name=True)

ENDPOINT_URL = "http://0.0.0.0:8080/infer" 
 
image_path = './pkrecog_031_0006.jpg'

output_path = "./sample_output"

if not os.path.exists(output_path):
  pl.Path(output_path).mkdir(parents=True, exist_ok=True)

def infer(): 
    image =np.asarray(Image.open(image_path)).astype(np.float32) 
    data = { 'image': image.tolist() } 
    response = requests.post(ENDPOINT_URL, json = data) 
    response.raise_for_status() 
    if response.ok:

        # get response values
        out = response.json()
        num_detections = out['num_detections']

        # parse values into np.arrays with correct
        # dimensions
        for key in out.keys():
            b  = out[key]
            print("{} is {}".format(key,type(out[key])))
            if type(b) == list:
                if len(b)>num_detections:
                    out[key] = np.array(b).reshape(num_detections,-1)
                else:
                    out[key] = np.array(b)
                print(out[key].shape)

        # reload image and draw boxes/labels
        image = cv2.imread(image_path)
        viz_utils.visualize_boxes_and_labels_on_image_array(
              image,
              out['detection_boxes'],
              out['detection_classes'],
              out['detection_scores'],
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.3,
              agnostic_mode=False)
 
        # save image
        plt.figure(figsize=(15,15))
        plt.imshow(image)
        cv2.imwrite(os.path.join(output_path, image_path.split("/")[-1]),image)

        
    else:
        print("Problem with response from server.")
if __name__ =="__main__": 
    infer() 