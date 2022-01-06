import cv2 as cv
import numpy as np
import sys
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from Code.IO.save_histograms import load_sift_h, save_sift_h, save_hog_h, load_hog_h, save_vgg_features, load_vgg_features, save_cnn, load_cnn
from Code.IO.load_data import FACE_WIDTH, FACE_HEIGHT
from Code.Data_Processing.create_train_data import create_train_data_facial
from skimage.feature import hog
from sklearn.svm import LinearSVC
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, backend, activations, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, Dropout, BatchNormalization
from tensorflow.python.client import device_lib
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG19, vgg19
from sklearn.model_selection import train_test_split


K = 50 # how many centroids to use in k-means
HOG_W = FACE_WIDTH / 4
HOG_H = FACE_HEIGHT / 4
CELLS_W = 1
CELLS_H = 1

def extract_features_facial_sift(positive_examples, negative_examples):
    if load_sift_h() is not None:
        return load_sift_h()

    sift = cv.SIFT_create(edgeThreshold=15, contrastThreshold=0.03)
    pos_kps = []
    for example in positive_examples:
        kp = sift.detect(example, None)
        pos_kps.append(kp)
    pos_kps, pos_dp = sift.compute(positive_examples, pos_kps)
    pos_dps = pos_dp[0]
    for dp in pos_dp:
        if dp is None:
            pos_dps = np.vstack((pos_dps, np.zeros((1, sift.descriptorSize()), np.float32)))
            continue
        pos_dps = np.vstack((pos_dps, dp))
    for index, dp in enumerate(pos_dp):
        if dp is None:
            pos_dp[index] = np.zeros((1, sift.descriptorSize()), np.float32)
    # pos_cb, variance = kmeans(pos_dps, K, 1)

    # do the same thing for negative examples
    neg_kps = []
    for example in negative_examples:
        kp = sift.detect(example, None)
        neg_kps.append(kp)
    neg_kps, neg_dp = sift.compute(negative_examples, neg_kps)
    neg_dps = neg_dp[0]
    for dp in neg_dp:
        if dp is None:
            neg_dps = np.vstack((neg_dps, np.zeros((1, sift.descriptorSize()), np.float32)))
            continue
        neg_dps = np.vstack((neg_dps, dp))
    for index, dp in enumerate(neg_dp):
        if dp is None:
            neg_dp[index] = np.zeros((1, sift.descriptorSize()), np.float32)

    complete_cb, variance = kmeans(np.concatenate((pos_dps, neg_dps)), K, 1)


    feature_histograms_neg = np.zeros((len(negative_examples), K), np.float32)
    feature_histograms_pos = np.zeros((len(positive_examples), K), np.float32)
    for index, example in enumerate(positive_examples):
        try:
            vwords, distances = vq(pos_dp[index], complete_cb)
        except ValueError:
            print("aaaa")
        for vw in vwords:
            feature_histograms_pos[index, vw] += 1

    for index, example in enumerate(negative_examples):
        vwords, distances = vq(neg_dp[index], complete_cb)
        for vw in vwords:
            feature_histograms_neg[index, vw] += 1

    save_sift_h(feature_histograms_pos, feature_histograms_neg, complete_cb)
    return feature_histograms_pos, feature_histograms_neg, complete_cb

def extract_VGG19_features_set(t_data):
    if load_vgg_features() is not None:
        return load_vgg_features()

    t_data = vgg19.preprocess_input(t_data)
    vgg = VGG19(include_top=False, input_shape=(FACE_HEIGHT, FACE_WIDTH, 3))
    # vgg.summary()
    stuff = vgg.predict(t_data)
    save_vgg_features(stuff)
    return stuff

def extract_sift_features_image(image):
    sift = cv.SIFT_create(edgeThreshold=15, contrastThreshold=0.03)
    kp = sift.detect(image, None)
    kps, dp = sift.compute(image, kp)
    return dp

def train_svm_facial(train_data, train_labels):
    svm = LinearSVC(C = 10 ** -2)
    svm.fit(train_data, train_labels)
    print(f"Score on train data {svm.score(train_data, train_labels)}")

    return svm


def train_cnn_facial(train_data, train_labels):
    if load_cnn() is not None:
        return load_cnn()
    cnn = models.Sequential()
    cnn.add(Flatten(input_shape=train_data[0].shape))
    cnn.add(Dense(100, activation='relu', kernel_initializer=GlorotNormal(), kernel_regularizer=regularizers.l2(1e-3)))
    cnn.add(Dropout(0.6))
    cnn.add(Dense(2, activation='softmax', kernel_initializer=GlorotNormal()))

    cnn.summary()
    cnn.compile(optimizer=SGD(learning_rate=1e-3, momentum=0.9, decay=1e-2 / 200),  # 200 = nr de epoci
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

    early = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=30, mode='auto', restore_best_weights=True)

    train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.15, stratify=train_labels)
    cnn.fit(train_data, train_labels, epochs=10, batch_size=16, callbacks=[early],validation_data=(valid_data, valid_labels), verbose=1)
    save_cnn(cnn)
    return cnn
