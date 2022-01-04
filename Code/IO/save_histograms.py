import numpy as np
import os
from Code.IO.load_data import DATA_PATH
S_POS_H = "sift_pos_hist"
S_NEG_H = "sift_neg_hist"
H_POS_H = "hog_pos_hist"
H_NEG_H = "hog_neg_hist"

def save_sift_h(pos_h, neg_h):
    np.save(DATA_PATH + S_POS_H + ".npy",pos_h)
    np.save(DATA_PATH + S_NEG_H + ".npy", neg_h)

def load_sift_h():
    pos_hist = os.path.join(DATA_PATH, S_POS_H + ".npy")
    if os.path.exists(pos_hist):
        pos_h = np.load(DATA_PATH + S_POS_H + ".npy")
        neg_h = np.load(DATA_PATH + S_NEG_H + ".npy")
        return pos_h, neg_h
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