import numpy as np
from Code.Logical.classes import FaceClasses
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

def create_train_data_facial(pos_ex, neg_ex):
    train_data = np.concatenate((pos_ex, neg_ex))
    train_labels = np.zeros(len(pos_ex) + len(neg_ex), np.uint8)
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

    return train_data
