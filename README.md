# 🌊 River Trash Detection System using YOLOv8

This repository contains the complete project developed from the uploaded notebook for detecting floating trash in river images using **YOLOv8**.

## 🎯 Project Objective

The system detects floating trash and related objects in river images using an object-detection model.

### Classes

The dataset configuration in the notebook contains 8 classes:

1. bottle
2. grass
3. branch
4. milk-box
5. plastic-bag
6. plastic-garbage
7. ball
8. leaf

## 📁 Repository Structure

```text
river-trash-detection/
│
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── data.yaml
├── river_trash_detection.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
└── best.pt                 # trained model; add separately if required
```

## 🧠 Training Configuration

The training section of the notebook uses:

- Model: YOLOv8n
- Epochs: 100
- Image size: 640
- Batch size: 8
- Patience: 25
- Horizontal flip probability: 0.5
- Dataset: River Floating Trash Datasets
- Dataset identifier: `zhiaun/river-floating-trash-datasets`

## 📊 Evaluation

`evaluate.py` reproduces the notebook's class-level metric calculations:

- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- Precision
- Recall
- Specificity
- F1 Score
- Accuracy

It also saves the calculated table to:

```text
metrics.csv
```

## 🔍 Prediction

`predict.py` loads `best.pt` and performs detection on:

```text
dataset/datasets/RFT/images/test
```

with a confidence threshold of `0.25`.

## 🌐 Web Application

`app.py` provides a Gradio interface containing:

- Login
- Sign Up
- Logout
- Image upload
- YOLOv8 trash detection
- Bounding-box output
- Detected class and confidence display
- SQLite user database

## ⚙️ Installation

Create a virtual environment if desired:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Train the Model

Download/extract the dataset and make sure its structure matches `data.yaml`.

Then:

```bash
python train.py
```

The trained weights will be created by Ultralytics under the training run directory.

Copy the required trained `best.pt` to the project root if you want to use the web application.

## ▶️ Evaluate the Model

Place your trained model at:

```text
best.pt
```

Then run:

```bash
python evaluate.py
```

## ▶️ Run Predictions

Place the test images in the dataset path described above and run:

```bash
python predict.py
```

## ▶️ Run the Web App

Place:

```text
best.pt
```

in the same directory as `app.py`.

Then:

```bash
python app.py
```

Open the Gradio URL shown in the terminal.

## 🔐 Authentication

The web application uses SQLite for local user accounts and SHA-256 hashing for the stored password value.

The generated database is:

```text
users.db
```

It is excluded from Git using `.gitignore`.

For a production deployment, use a modern password-hashing algorithm such as Argon2 or bcrypt and a proper session/authentication system.

## 🔑 Kaggle Credentials

The original notebook uses `kaggle.json` to download the dataset.

**Do not upload `kaggle.json` to GitHub.**

Keep it local and configure the Kaggle API separately if you need to download the dataset.

## 📓 Original Notebook

`river_trash_detection.ipynb` is the original uploaded notebook, preserved as a project reference.

It contains the workflow for:

- Installing Ultralytics
- Dataset download
- Dataset extraction
- YOLOv8 training
- Model saving
- Validation
- Confusion matrix
- Class-wise metrics
- Performance visualizations
- Test-image prediction
- Gradio application

## ⚠️ Large Model Files

Trained `.pt` files can be large, so they are ignored by `.gitignore`.

If your GitHub repository needs to contain the trained model, consider Git LFS or a suitable model-hosting service.

## 📌 Dataset

The notebook references:

**River Floating Trash Datasets**

Kaggle identifier:

```text
zhiaun/river-floating-trash-datasets
```

Check the dataset license before redistributing the dataset or its images.

## 👩‍💻 Project Title

**River Trash Detection System using YOLOv8**

A computer-vision project for detecting floating waste in water bodies.
