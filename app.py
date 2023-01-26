import rasterio
from rasterio.enums import Resampling
from PIL import Image
import requests

from flask import Flask, request,send_file
import numpy as np
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

def url_to_image (link):
    response = requests.get(link,timeout=99999999)
    with open('sar.tif', 'wb') as f:
        f.write(response.content)

def load_images_sar(filename):
    with rasterio.open(filename) as src:
          img_data = src.read(
            out_shape=(src.count,400,400),
            resampling=Resampling.bilinear )
          img_data = img_data[:3]
          img_data = np.moveaxis(img_data, 0,-1) 
    return img_data

app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    url = request.json["url"]
    url_to_image(url)
    cust = {'InstanceNormalization': InstanceNormalization}
    model ='./g_model_AtoB_003600.h5'
    model_AtoB = load_model(model, cust)   
    opt = Adam(lr=0.0002, beta_1=0.5)
    model_AtoB.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    sar_img = load_images_sar('sar.tif')
    sar_img = np.expand_dims(sar_img,axis=0)
    generatedImage = model_AtoB.predict(sar_img)
    generatedImage = (generatedImage + 1) / 2.0
    image_data = (generatedImage[0] * 255).astype(np.uint8)
    image = Image.fromarray(np.uint8(image_data))
    image.save('generatedImage.jpg')
    print("aman",request)
    return send_file('generatedImage.jpg', mimetype='image/jpg')

if __name__=="__main__":
    app.run(debug=True,port="7910")