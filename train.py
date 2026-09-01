from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    patience=25,
    fliplr=0.5
)

model.save("my_model.pt")
print("Model saved successfully!")
