"""
author: Apr0G & N1chey
"""

X = "Oil Spills"
Y = "Wildfire"
Z = "Neither"

TRAIN_DATA_DIR = 'train'

# How many times wider than the training cluster radius before we call it "Neither".
# Lower = more sensitive (more images become Neither). Start at 1.5 and tune.
NEITHER_SENSITIVITY = 1.5

# If similarity to the nearest class is below this, also call it "Neither".
# 0.4 means "less than 40% similar → Neither".
MIN_CONFIDENCE = 0.4

sampleX = 'static/OilSpills.jpeg'
sampleY = 'static/ForrestFire.jpeg'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ML_MODEL_FILENAME = 'saved_model.h5'

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image
import keras

app = Flask(__name__)


def load_image(path):
    img = Image.open(path).resize((150, 150)).convert('RGB')
    return np.array(img, dtype='float32') / 255.0


def build_feature_extractor(model):
    # Use the Dense(64) layer output — second-to-last trainable layer,
    # before dropout and the final sigmoid. Gives a 64-dim feature vector.
    dense64_layer = None
    for layer in model.layers:
        if hasattr(layer, 'units') and layer.units == 64:
            dense64_layer = layer
    if dense64_layer is None:
        raise RuntimeError("Could not find Dense(64) layer in model")
    return keras.Model(inputs=model.inputs, outputs=dense64_layer.output)


def compute_centroids_and_radius(feature_extractor, train_dir):
    """
    Returns:
      centroids: {class_index: mean_feature_vector}
      radius: max distance from any training image to its own centroid
              (defines the boundary of "known" space)
    """
    class_features = {}
    for class_idx, class_name in enumerate(sorted(os.listdir(train_dir))):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        features = []
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            try:
                arr = np.expand_dims(load_image(fpath), axis=0)
                feat = feature_extractor.predict(arr, verbose=0)[0]
                features.append(feat)
            except Exception:
                continue
        if features:
            class_features[class_idx] = np.array(features)

    centroids = {cls: feats.mean(axis=0) for cls, feats in class_features.items()}

    # Radius = largest distance from any training image to its own centroid
    radius = 0.0
    for cls, feats in class_features.items():
        centroid = centroids[cls]
        for feat in feats:
            dist = np.linalg.norm(feat - centroid)
            if dist > radius:
                radius = dist

    return centroids, radius


def classify(feature_extractor, centroids, radius, test_image):
    feat = feature_extractor.predict(test_image, verbose=0)[0]
    distances = {cls: np.linalg.norm(feat - centroid) for cls, centroid in centroids.items()}
    nearest_cls = min(distances, key=distances.get)
    nearest_dist = distances[nearest_cls]
    threshold = radius * NEITHER_SENSITIVITY

    confidence = 1.0 - (nearest_dist / threshold)
    if nearest_dist > threshold or confidence < MIN_CONFIDENCE:
        return Z, None
    elif nearest_cls == 0:
        return X, confidence
    else:
        return Y, confidence


def load_model_from_file():
    return keras.saving.load_model(ML_MODEL_FILENAME)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html', myX=X, myY=Y, myZ=Z, mySampleX=sampleX, mySampleY=sampleY)
    else:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('I only accept files of type ' + str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    test_image = np.expand_dims(load_image(filepath), axis=0)

    feature_extractor = app.config['FEATURE_EXTRACTOR']
    centroids = app.config['CENTROIDS']
    radius = app.config['RADIUS']

    label, confidence = classify(feature_extractor, centroids, radius, test_image)
    image_src = "/" + UPLOAD_FOLDER + "/" + filename

    if label == X:
        conf_str = f"{confidence*100:.0f}% similar to training data"
        answer = f"<div class='col text-center'><img width='150' height='150' src='{image_src}' class='img-thumbnail' /><h4>guess: {X}<br/><small>{conf_str}</small></h4></div><div class='col'></div><div class='col'></div><div class='w-100'></div>"
    elif label == Y:
        conf_str = f"{confidence*100:.0f}% similar to training data"
        answer = f"<div class='col'></div><div class='col text-center'><img width='150' height='150' src='{image_src}' class='img-thumbnail' /><h4>guess: {Y}<br/><small>{conf_str}</small></h4></div><div class='col'></div><div class='w-100'></div>"
    else:
        answer = f"<div class='col'></div><div class='col'></div><div class='col text-center'><img width='150' height='150' src='{image_src}' class='img-thumbnail' /><h4>guess: {Z}</h4></div><div class='w-100'></div>"

    results.append(answer)
    return render_template('index.html', myX=X, myY=Y, myZ=Z, mySampleX=sampleX, mySampleY=sampleY, len=len(results), results=results)


def main():
    print("Loading model...")
    myModel = load_model_from_file()

    print("Building feature extractor...")
    feature_extractor = build_feature_extractor(myModel)

    print("Computing training centroids...")
    centroids, radius = compute_centroids_and_radius(feature_extractor, TRAIN_DATA_DIR)
    print(f"Cluster radius: {radius:.4f}  |  Neither threshold: {radius * NEITHER_SENSITIVITY:.4f}")

    app.config['SECRET_KEY'] = 'super secret key'
    app.config['FEATURE_EXTRACTOR'] = feature_extractor
    app.config['CENTROIDS'] = centroids
    app.config['RADIUS'] = radius
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(port=5001)


results = []

main()
