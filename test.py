from PIL import Image 
import numpy as np 
from inferer import Inferer 
 
class MyTestCase(unittest.TestCase): 
	def test_infer(self): 
    	image = np.asarray(Image.open('resources/yorkshire_terrier.jpg')).astype(np.float32) 
        inferrer = Inferer() 
        inferrer.infer(image) 
