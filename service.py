
from flask import Flask, request, jsonify
from inferer import Inferer
import os
import traceback
import numpy as np
app = Flask(__name__) 


APP_ROOT = os.getenv('APP_ROOT', '/infer')  
print(" ****  {} **** ".format(APP_ROOT))
HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv('PORT_NUMBER',8080))

u_net = Inferer() 


quarks = [{'name': 'up', 'charge': '+2/3'},
          {'name': 'down', 'charge': '-1/3'},
          {'name': 'charm', 'charge': '+2/3'},
          {'name': 'strange', 'charge': '-1/3'}]

@app.route(APP_ROOT, methods=["POST"]) 
def infer(): 
	data = request.get_json() 
	image = data['image'] 
	out = u_net.infer(image) 
	print("OUTPUT: \n {}".format(out))
	for key in out.keys():
		b = out[key]
		print("{} is {}".format(key,type(out[key])))
		if type(b) == np.ndarray:
			print(b.shape)
			out[key] = b.tolist()
	return jsonify(out)

@app.errorhandler(Exception) 
def handle_exception(e): 
	return jsonify(stackTrace=traceback.format_exc()) 


if __name__ == '__main__': 
    app.run(debug=True, host=HOST, port=PORT_NUMBER) #host=HOST
