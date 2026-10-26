from ultralytics import YOLO

model = YOLO("yolov8n.pt")
CONFIG_FILE = "dataset/data.yaml"

# baseline training
model.train(
    data=CONFIG_FILE,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    name="office_items_detection",
    seed=42,
)