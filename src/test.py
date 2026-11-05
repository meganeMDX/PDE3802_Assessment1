from ultralytics import YOLO
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "best.pt"
CONFIG_FILE = ROOT / "dataset" / "data.yaml"

model = YOLO(MODEL_PATH)

results = model.val(
    data=CONFIG_FILE,
    split="test",
    imgsz=640
)

# Overall metrics
print("\nmAP50-95:", round(results.box.map, 3))  
print("mAP50:", round(results.box.map50, 3))  
print("mAP75:", round(results.box.map75, 3))  
print("Mean precision:", round(results.box.mp, 3))
print("Mean recall:", round(results.box.mr, 3))

# for confusion matrix, macro metrics and accuracy
results_cm = model.val(
    data=CONFIG_FILE,
    split="test",
    imgsz=640,
    conf=0.25,
    iou=0.45,
    verbose=False
)

cm = results_cm.confusion_matrix.matrix
nc = len(results_cm.names)
print("\nClass names:", results_cm.names)

# calculate macro-precision, macro-recall and macro-F1
# Adapted from glenn-jocher (2024), Ultralytics GitHub issue #9946
# "YOLOv8 classification metrics"
# Available at: https://github.com/ultralytics/ultralytics/issues/9946#issuecomment-2132207985
TP = np.diag(cm)[:nc]
FP = cm.sum(axis=1)[:nc] - TP
FN = cm.sum(axis=0)[:nc] - TP

# metrics for each class
precision = np.divide(TP, TP + FP, out=np.zeros(nc), where=(TP + FP) > 0) 
recall = np.divide(TP, TP + FN, out=np.zeros(nc), where=(TP + FN) > 0) 
f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(nc), where=(precision + recall) > 0) 

for i in range(nc):
    print(f"{results_cm.names[i]:15s} P={precision[i]:.3f} R={recall[i]:.3f} F1={f1[i]:.3f}")

precision_overall = np.mean(precision)
recall_overall = np.mean(recall)
f1_overall = np.mean(f1)

print(f"\nMacro-Precision: {round(precision_overall, 3)}")
print(f"Macro-Recall: {round(recall_overall, 3)}")
print(f"Macro-F1: {round(f1_overall, 3)}")