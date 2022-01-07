from Code.IO.load_data import get_examples, get_valid, FACE_HEIGHT, FACE_WIDTH
from Code.Model.cnn_bow import extract_features_facial_sift, train_svm_facial, train_cnn_facial, extract_VGG19_features_set, extract_VGG19_features_augmented
from Code.Data_Processing.create_train_data import create_train_data_facial, normalize_train_data, create_valid_data
from Code.Model.sliding_window import sliding_window_valid, check_detections_directly
from Code.Model.evaluate_results import evaluate_detections_facial
from sklearn.utils import shuffle
from Code.Data_Processing.augmentation import augment_facial_data
import numpy as np
import cv2 as cv
from PIL import Image
import time
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

def main():
    faces, faces_coords, negatives = get_examples()
    hsum, wsum = 0, 0
    valid, valid_labels = get_valid()
    t_data, t_labels = create_train_data_facial(faces, negatives)
    # t_data, t_labels = create_train_data_facial(faces, negatives)
    # for train in t_data:
    #     c_train = array_to_img(train)
    #     c_train.show()
    #     time.sleep(1)
    features = extract_VGG19_features_set(t_data)
    # s_pos_h, s_neg_h, s_cb = extract_features_facial_sift(faces, negatives)
    # t_data, t_labels = create_train_data_facial(s_pos_h, s_neg_h[np.random.choice(len(s_neg_h), size=30000)])
    # features = features.reshape(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])
    # d_pos, d_neg = augment_facial_data(faces, negatives)
    # features_pos, features_neg = extract_VGG19_features_augmented(d_pos, d_neg, len(faces), len(negatives))
    old_shape = features.shape
    features, scaler = normalize_train_data(features.reshape(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))
    features = features.reshape(*old_shape)
    # svm = train_svm_facial(features, t_labels)
    cnn = train_cnn_facial(features, t_labels)
    print("stuff")
    # valid, valid_labels = create_valid_data(valid, valid_labels)
    # check_detections_directly(valid, valid_labels, None, cnn, scaler)
    new_valid = shuffle(valid)
    detections = sliding_window_valid(valid[:5], cnn)
    evaluate_detections_facial(detections, valid_labels, valid[:5])

if __name__ == "__main__":
    main()