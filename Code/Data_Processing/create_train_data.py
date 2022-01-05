import numpy as np
import cv2 as cv
import time
from Code.Logical.classes import FaceClasses
from Code.IO.load_data import FACE_WIDTH, FACE_HEIGHT
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

def create_train_data_facial(pos_ex, neg_ex):
    train_data = np.concatenate((pos_ex, neg_ex))
    train_labels = np.zeros(len(pos_ex) + len(neg_ex), np.int8)
    for i in range(len(pos_ex)):
        train_labels[i] = FaceClasses.Face.value
    for i in range(len(neg_ex)):
        train_labels[i + len(pos_ex)] = FaceClasses.NoFace.value

    train_data, train_labels = shuffle(train_data, train_labels)
    return train_data, train_labels

def normalize_train_data(train_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    return train_data, scaler

def create_valid_data(valid_raw, valid_labels):
    valid = []
    vl = []
    for im_index, image in enumerate(valid_raw):
        for l_index, label in enumerate(valid_labels):
            vi_index, x1, y1, x2, y2, im_class = label
            # print(vi_index, im_index)
            if vi_index < im_index:
                continue
            if vi_index > im_index:
                break
            face = image[y1:y2, x1:x2]
            face = img_to_array(array_to_img(face).resize((FACE_WIDTH, FACE_HEIGHT)))
            # array_to_img(face).show()
            # time.sleep(1)
            valid.append(face)
            vl.append(FaceClasses.Face.value)
    valid = np.asarray(valid)
    vl = np.asarray(vl)
    return valid, vl

