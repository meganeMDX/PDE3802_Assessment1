from ultralytics import YOLO
import numpy as np
from IPython.display import Image, display

MODEL_PATH = "runs/detect/office_items_detection/weights/best.pt"
CONFIG_FILE = "dataset/data.yaml"

model = YOLO(MODEL_PATH)

results = model.val(
    data=CONFIG_FILE,
    split="test",
    imgsz=640
)

# Overall metrics
print("\nmAP50-95:", round(results.box.map, 2))  
print("mAP50:", round(results.box.map50, 2))  
print("mAP75:", round(results.box.map75, 2))  
print("Mean precision:", round(results.box.mp, 2))
print("Mean recall:", round(results.box.mr, 2))

# Per-class metrics
print("\nClass indices evaluated:", results.box.ap_class_index)
print("Per-class mAP50-95:", results.box.maps)
    
cm = results.confusion_matrix.matrix
nc = len(results.names)
print("\nClass names:", results.names)
print("Confusion matrix shape:", cm.shape)
print(cm)

# calculate accuracy, precision, recall and F1
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
    print(f"{results.names[i]:15s} P={precision[i]:.2f} R={recall[i]:.2f} F1={f1[i]:.2f}")

precision_overall = np.mean(precision)
recall_overall = np.mean(recall)
f1_overall = np.mean(f1)
accuracy = TP.sum()/ cm.sum()

print(f"\nPrecision: {round(precision_overall, 2)}")
print(f"Recall: {round(recall_overall, 2)}")
print(f"Macro-F1: {round(f1_overall, 2)}")
print(f"Accuracy: {round(accuracy, 2)}")

graphs = [
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "val_batch0_pred.jpg" 
]
for graph in graphs:
    display(Image(filename=f"{results.save_dir}/{graph}"))