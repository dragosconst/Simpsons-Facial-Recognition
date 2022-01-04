import cv2 as cv
import numpy as np
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from Code.IO.save_histograms import load_sift_h, save_sift_h, save_hog_h, load_hog_h
from Code.IO.load_data import FACE_WIDTH, FACE_HEIGHT
from skimage.feature import hog
from sklearn.svm import LinearSVC

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

def extract_sift_features_image(image):
    sift = cv.SIFT_create(edgeThreshold=15, contrastThreshold=0.03)
    kp = sift.detect(image, None)
    kps, dp = sift.compute(image, kp)
    if dp is None:
        return np.zeros((1, sift.descriptorSize()), np.float32)
    return dp

def train_svm_facial(train_data, train_labels):
    svm = LinearSVC(C = 10 ** -2)
    svm.fit(train_data, train_labels)
    print(f"Score on train data {svm.score(train_data, train_labels)}")

    return svm

