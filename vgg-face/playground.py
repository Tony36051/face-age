from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Convolution Features
vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max

# After this point you can use your model to predict.
img_path = r'D:\aligned\imdb_crop\00\nm0000100_rm96704768_1955-1-6_2003.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
model = vgg_features
features = model.predict(x)