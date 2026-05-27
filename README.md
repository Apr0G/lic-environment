# lic-environment

A Flask web app that classifies drone images as either an oil spill or a wildfire using a custom-trained CNN. Upload an image and get an instant prediction — every upload is saved to continuously expand the training dataset.

---

## How it works

**Model (`trainTheBrain.py`)** — a binary CNN built with Keras: three Conv2D + MaxPooling blocks, a fully connected layer with 0.5 Dropout, and a sigmoid output. Trained on images in `train/` and evaluated against `validation/`, then saved as `saved_model.h5`.

**Web app (`website.py`)** — Flask serves an image upload form. Submitted images are resized to 150×150, passed through the loaded model, and classified as either oil spill or wildfire with a confidence score.

**Data flywheel** — all uploaded images are stored in `static/uploads/`, building up a dataset for future retraining runs.

---

## Stack

Python · Flask · TensorFlow / Keras · NumPy · Pillow

---

## Running

**1. Prepare your dataset**

Place training images in:
```
train/
  OilSpills/
  Wildfire/
validation/
  OilSpills/
  Wildfire/
```

> For better model performance, use a large and diverse dataset. Good sources: [Kaggle wildfire datasets](https://www.kaggle.com/search?q=wildfire), [oil spill satellite imagery](https://www.kaggle.com/search?q=oil+spill+detection). The more images per class, the more reliable the predictions.

**2. Train the model:**

```bash
python trainTheBrain.py
```

**3. Launch the web app:**

```bash
python website.py
```

Open `http://localhost:5000`, upload a drone image, and get a prediction.

---

## Built by

[Faig Safiyev](https://github.com/Apr0G) & [n1chey](https://github.com/n1chey)
