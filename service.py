
import flask
from flask import Flask, request, jsonify
from inferer import Inferer
import os
import traceback
import numpy as np
import cv2
import werkzeug
app = Flask(__name__) 
app.config['JSON_AS_ASCII'] = False


APP_ROOT = os.getenv('APP_ROOT', '/infer')  
print(" ****  {} **** ".format(APP_ROOT))
HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv('PORT_NUMBER',5000))

u_net = Inferer() 


@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = u_net.infer(img) 

    return jsonify(out)


@app.route(APP_ROOT, methods=["POST"]) 
def infer(): 
    data = request.get_json() 
    image = np.array(data['image']).reshape(data['height'], data['width'],-1).astype(np.uint8) 
    print("===== TYPE: ",type(image), image.shape)
    out = u_net.infer(image) 
    print("OUTPUT: \n {}".format(out))
    #for key in out.keys():
    #    b = out[key]
    #    print("{} is {}".format(key,type(out[key])))
    #    if type(b) == np.ndarray:
    #        print(b.shape)
    #        out[key] = b.tolist()
    return out#jsonify(out)

@app.errorhandler(Exception) 
def handle_exception(e): 
    return jsonify(stackTrace=traceback.format_exc()) 


if __name__ == '__main__': 
    app.run(debug=True, host=HOST, port=PORT_NUMBER) #host=HOST
