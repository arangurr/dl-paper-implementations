import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from skimage import img_as_ubyte
from skimage.color import lab2rgb, rgb2lab
from skimage.transform import resize
import argparse

def recolorize_img(img):
    h,w = 128,128
    lab = np.asarray(rgb2lab(resize(img, (h,w))))
    x_black = lab[:,:,:1]/100.
    y_pred = model.predict(x_black.reshape(1,128,128,1))

    res_img = np.empty((h,w,3))
    res_img[:,:,:1] = x_black*100
    res_img[:,:,1:] = y_pred*128

    rgb = lab2rgb(res_img)
    return img_as_ubyte(rgb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Colorization')
    parser.add_argument('--model', default="../data/faces.h5",
                        type=str, help='Keras model to load')
    parser.add_argument('--img', default="../data/image00002.jpg",
                        type=str, help='Image to colorize')

    args = parser.parse_args()

    model_path = args.model
    img_path = args.img

    model = load_model(model_path)

    img = img_to_array(load_img(img_path)).astype('uint8')
    col = recolorize_img(img)

    plt.imshow(col)
    plt.title("colorized img")
    plt.show()
    plt.imshow(img)
    plt.title("original img")
    plt.show()