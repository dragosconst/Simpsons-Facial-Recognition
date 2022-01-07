import numpy as np
import cv2 as cv
import time
from Code.Logical.classes import FaceClasses, ImageClasses
from Code.IO.load_data import FACE_WIDTH, FACE_HEIGHT
from Code.IO.load_data import NAMES
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

def create_train_data_face_classes(faces, faces_labels):
    t_data, t_labels = shuffle(faces, faces_labels)
    return t_data, t_labels

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

def check_valid_labels(v_labels):
    v_labels = list(v_labels)
    for index, label in enumerate(v_labels):
        f_index, im_class = label
        if im_class in np.array([ImageClasses.Bart.value, ImageClasses.Homer.value, ImageClasses.Lisa.value, ImageClasses.Marge.value, ImageClasses.Unknown.value]):
            continue
        if im_class[:4] == "bart" or im_class == str(ImageClasses.Bart.value):
            v_labels[index] = ImageClasses.Bart.value
        elif im_class[:5] == "homer" or im_class == str(ImageClasses.Homer.value):
            v_labels[index] = ImageClasses.Homer.value
        elif im_class[:4] == "lisa" or im_class == str(ImageClasses.Lisa.value):
            v_labels[index] = ImageClasses.Lisa.value
        elif im_class[:5] == "marge" or im_class == str(ImageClasses.Marge.value):
            v_labels[index] = ImageClasses.Marge.value
        elif im_class[:7] == "unknown" or im_class == str(ImageClasses.Unknown.value):
            v_labels[index] = ImageClasses.Unknown.value
    return np.asarray(v_labels)