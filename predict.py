from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "best.pt"
TEST_DIR = "dataset/datasets/RFT/images/test"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=TEST_DIR,
    conf=0.25,
    save=True,
    save_txt=False
)

print("Prediction completed.")
