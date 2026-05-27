"""
author: AproG123 & N1chey
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers
import numpy as np
from PIL import Image

IMG_WIDTH, IMG_HEIGHT = 150, 150
TRAIN_DATA_DIR = 'train'
VALIDATION_DATA_DIR = 'validation'

EPOCHS = 100
BATCH_SIZE = 8
ML_MODEL_FILENAME = 'saved_model.h5'
AUGMENT_FACTOR = 10   # generate this many augmented copies per original image


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax'),
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    return model


def augment(arr):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    # Random horizontal flip
    if np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random vertical flip
    if np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # Random rotation
    angle = np.random.uniform(-30, 30)
    img = img.rotate(angle)
    # Random brightness
    arr = np.array(img, dtype='float32') / 255.0
    arr = np.clip(arr * np.random.uniform(0.7, 1.3), 0.0, 1.0)
    return arr


def load_images(directory):
    images, labels = [], []
    class_dirs = sorted(d for d in os.listdir(directory)
                        if os.path.isdir(os.path.join(directory, d)))
    for label, class_name in enumerate(class_dirs):
        class_path = os.path.join(directory, class_name)
        for fname in sorted(os.listdir(class_path)):
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).resize((IMG_WIDTH, IMG_HEIGHT)).convert('RGB')
                images.append(np.array(img, dtype='float32') / 255.0)
                labels.append(label)
            except Exception:
                continue
    return np.array(images), np.array(labels)


def make_augmented_dataset(images, labels, factor):
    aug_images = [images]
    aug_labels = [labels]
    for _ in range(factor - 1):
        aug_images.append(np.array([augment(img) for img in images]))
        aug_labels.append(labels)
    X = np.concatenate(aug_images, axis=0)
    y = np.concatenate(aug_labels, axis=0)
    shuffle = np.random.permutation(len(X))
    return X[shuffle], y[shuffle]


def main():
    print("Loading data...")
    train_images, train_labels = load_images(TRAIN_DATA_DIR)
    val_images, val_labels = load_images(VALIDATION_DATA_DIR)
    print(f"Original — Train: {len(train_images)}  Val: {len(val_images)}")

    print(f"Augmenting training set {AUGMENT_FACTOR}x...")
    train_images_aug, train_labels_aug = make_augmented_dataset(train_images, train_labels, AUGMENT_FACTOR)
    print(f"Augmented train size: {len(train_images_aug)}")

    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            ML_MODEL_FILENAME, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    model.fit(
        train_images_aug, train_labels_aug,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
    )

    print("Training complete. Best model saved.")


main()
