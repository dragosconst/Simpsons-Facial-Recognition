import cv2 as cv
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import numpy as np

LY = (12, 186, 134)
HY = (141, 206, 186)

def apply_filters(image_array):
    im_pil = array_to_img(image_array)
    image = cv.cvtColor(np.array(im_pil), cv.COLOR_RGB2BGR)
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(image_hsv, LY, HY)
    image = cv.bitwise_and(image, image, mask=mask)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = img_to_array(array_to_img(image))
    return image
