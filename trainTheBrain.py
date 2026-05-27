"""
author: AproG123 & N1chey
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_WIDTH, IMG_HEIGHT = 150, 150
TRAIN_DATA_DIR = 'train'
VALIDATION_DATA_DIR = 'validation'

EPOCHS = 50
BATCH_SIZE = 5
ML_MODEL_FILENAME = 'saved_model.h5'


def build_model():
    model = keras.Sequential()
    model.add(layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def load_images_from_directory(directory, target_size):
    images = []
    labels = []
    class_dirs = sorted(os.listdir(directory))
    for label, class_dir in enumerate(class_dirs):
        class_path = os.path.join(directory, class_dir)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).resize(target_size)
                img = img.convert('RGB')
                arr = np.array(img, dtype='float32') / 255.0
                images.append(arr)
                labels.append(label)
            except Exception:
                continue
    return np.array(images), np.array(labels)


def train_model(model):
    train_images, train_labels = load_images_from_directory(
        TRAIN_DATA_DIR, (IMG_WIDTH, IMG_HEIGHT))
    val_images, val_labels = load_images_from_directory(
        VALIDATION_DATA_DIR, (IMG_WIDTH, IMG_HEIGHT))

    print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")

    model.fit(
        train_images, train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_images, val_labels))

    return model


def main():
    myModel = build_model()
    myModel = train_model(myModel)
    myModel.save(ML_MODEL_FILENAME)


main()
