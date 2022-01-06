from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

EXTRA_POS = 10 ** 4
EXTRA_NEG = 2 * 10 ** 4
BATCH = 32

def augment_facial_data(positives, negatives):
    datagen = ImageDataGenerator(rotation_range=90,
                                 shear_range=0.2, vertical_flip=True, fill_mode='nearest')
    dataset_pos = datagen.flow(positives, np.zeros(len(positives)), batch_size=BATCH)
    dataset_neg = datagen.flow(negatives, np.ones(len(negatives)), batch_size=BATCH)
    return dataset_pos, dataset_neg

