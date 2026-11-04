from ultralytics import YOLO
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "best.pt"
CONFIG_FILE = ROOT / "dataset" / "own_images" / "data.yaml"
IMG_DIR = ROOT / "dataset" / "own_images" / "test" / "images"
SAVE_DIR = ROOT / "own_images_predictions"

model = YOLO(MODEL_PATH)

# evaluate model on own images
results = model.val(
    data=CONFIG_FILE,
    split="test",
    imgsz=640,
    plots=False
)

print("\nOwn Images Evaluation:")
print("mAP50-95:", round(results.box.map, 3))  
print("mAP50:", round(results.box.map50, 3))  
print("mAP75:", round(results.box.map75, 3))  
print("Mean precision:", round(results.box.mp, 3))
print("Mean recall:", round(results.box.mr, 3))

if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR, ignore_errors=True)

# save annotated predictions
result = model.predict(
    source=IMG_DIR,
    imgsz=640,
    conf=0.25,
    save=True,
    project=SAVE_DIR.parent,
    name=SAVE_DIR.name
)