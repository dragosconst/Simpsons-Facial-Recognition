from tensorflow.keras.applications import VGG19, vgg19
from Code.Model.cnn_bow import extract_VGG19_features_set
from Code.IO.save_histograms import save_vgg_mlp, load_vgg_mlp
from Code.Data_Processing.create_train_data import normalize_train_data
from tensorflow.keras import datasets, layers, models, backend, activations, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, Dropout, BatchNormalization
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

# first attempt, classify based on vgg extracted features
def classify_vgg19_features_facial(t_data, t_labels):
    if load_vgg_mlp() is not None:
        return load_vgg_mlp()

    features = extract_VGG19_features_set(t_data, faces_classes=True)
    old_shape = features.shape
    features, scaler = normalize_train_data(features.reshape(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))
    features = features.reshape(*old_shape)


    mlp = models.Sequential()
    mlp.add(Flatten(input_shape=features.shape[1:]))
    mlp.add(Dense(100, activation='relu', kernel_initializer=GlorotNormal(), kernel_regularizer=regularizers.l2(1e-3)))
    mlp.add(Dropout(0.6))
    mlp.add(Dense(5, activation='softmax', kernel_initializer=GlorotNormal()))

    mlp.summary()
    mlp.compile(optimizer=SGD(learning_rate=1e-3, momentum=0.9, decay=1e-2 / 200),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

    early = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=30, mode='auto', restore_best_weights=True)

    features, v_data, t_labels, v_labels = train_test_split(features, t_labels, test_size=0.1, stratify=t_labels)

    mlp.fit(features, t_labels, epochs=50, validation_data=(v_data, v_labels), batch_size=16, callbacks=[early], verbose=1)

    save_vgg_mlp(mlp)
    return mlp
