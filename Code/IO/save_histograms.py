import numpy as np
import os
import pickle
from Code.IO.load_data import DATA_PATH
from tensorflow.keras.models import load_model

S_POS_H = "sift_pos_hist"
S_NEG_H = "sift_neg_hist"
S_CB = "sift_cb"
H_POS_H = "hog_pos_hist"
H_NEG_H = "hog_neg_hist"
VGG_FEATURES = "vgg_features"
VGG_FACES = "vgg_faces_features"
VGG_MLP = "vgg_mlp"
AUG_POS = "aug_pos"
AUG_NEG = "aug_neg"

def save_sift_h(pos_h, neg_h, s_cb):
    np.save(DATA_PATH + S_POS_H + ".npy",pos_h)
    np.save(DATA_PATH + S_NEG_H + ".npy", neg_h)
    np.save(DATA_PATH + S_CB + ".npy", s_cb)

def load_sift_h():
    pos_hist = os.path.join(DATA_PATH, S_CB + ".npy")
    if os.path.exists(pos_hist):
        pos_h = np.load(DATA_PATH + S_POS_H + ".npy")
        neg_h = np.load(DATA_PATH + S_NEG_H + ".npy")
        s_cb = np.load(DATA_PATH + S_CB + ".npy")
        return pos_h, neg_h, s_cb
    return None

def save_hog_h(pos_h, neg_h):
    np.save(DATA_PATH + H_POS_H + ".npy", pos_h)
    np.save(DATA_PATH + H_NEG_H + ".npy", neg_h)

def load_hog_h():
    pos_hist = os.path.join(DATA_PATH, H_POS_H + ".npy")
    if os.path.exists(pos_hist):
        pos_h = np.load(DATA_PATH + H_POS_H + ".npy")
        neg_h = np.load(DATA_PATH + H_NEG_H + ".npy")
        return pos_h, neg_h
    return None

def save_vgg_features(features):
    np.save(os.path.join(DATA_PATH, VGG_FEATURES + ".npy"), features)

def load_vgg_features():
    ft_path = os.path.join(DATA_PATH, VGG_FEATURES +".npy")
    if os.path.exists(ft_path):
        features = np.load(ft_path)
        return features
    return None

def save_vgg_faces_features(features):
    np.save(os.path.join(DATA_PATH, VGG_FACES + ".npy"), features)

def load_vgg_faces_features():
    ft_path = os.path.join(DATA_PATH, VGG_FACES + ".npy")
    if os.path.exists(ft_path):
        features = np.load(ft_path)
        return features
    return None

def save_vgg_mlp(mlp):
    mlp.save(DATA_PATH + VGG_MLP)

def load_vgg_mlp():
    mlp_path = os.path.join(DATA_PATH, VGG_MLP)
    if os.path.exists(mlp_path):
        return load_model(mlp_path)
    return None

def save_cnn(cnn):
    # pickle.dump(cnn, open(DATA_PATH + 'cnn', 'wb'))
    cnn.save(DATA_PATH + 'cnn.cnn')

def load_cnn():
    cnn_path = os.path.join(DATA_PATH, 'cnn.cnn')
    if os.path.exists(cnn_path):
        return load_model(cnn_path)
    return None

def save_augmented_features(features_pos, features_neg):
    np.save(os.path.join(DATA_PATH, AUG_POS + ".npy"), features_pos)
    np.save(os.path.join(DATA_PATH, AUG_NEG + ".npy"), features_neg)

def load_augmented_features():
    ft_path = os.path.join(DATA_PATH, AUG_POS + ".npy")
    neg_path = os.path.join(DATA_PATH, AUG_NEG + ".npy")
    if os.path.exists(ft_path):
        return np.load(ft_path), np.load(neg_path)
    return None