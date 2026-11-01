import cv2, numpy as np, time
from pathlib import Path
from ultralytics import YOLO

modelName= "yolo11n"
imgSize=640
conf_thr = 0.5
nms_thr = 0.4
saveAnnotations = True
root = Path(__file__).resolve().parent.parent
Img =  root/"dataset"/ "processed"/ "test" /"images"
labels = root/"dataset"/ "processed"/ "test" /"labels"
weights = root/"runs/detect/office_items_detection/weights/best.pt"

#function to load the configuration for yolov8
def load_net(weights=str(weights)):
    return YOLO(weights)

net= load_net()
#testing to see if all the classes are well chosen
print("model load net:", net.names)

#obtaining the classes names from trained model to match ids
classes=[net.names[k] for k in sorted(net.names)]
#colours 
colours= np.random.default_rng(42).integers(0, 255, size=(len(classes), 3)).tolist()

#running thresholding before and nnms is applied
def detect(net, img, size =416, conf_thr=0.5):
    """Run YOLO and return every box above conf_thr (BEFORE NMS)."""
    start = time.time()
    H, W = img.shape[:2]
    outputs = net.predict(img, imgsz=size,conf=conf_thr, iou=1.0, verbose=False)[0]
    elapsed = time.time() - start

    #rearranged this section
    boxes, scores, class_ids, heads = [], [], [], []
    for row, score, cid in zip(outputs.boxes.xywhn.tolist(), outputs.boxes.conf.tolist(),outputs.boxes.cls.tolist()):
        if score > conf_thr:
            cx, cy, w, h = np.array(row * np.array([W,H,W,H]))
            boxes.append([int(cx - w/2), int(cy -h/2), int(w), int(h)])
            scores.append(float(score))
            class_ids.append(int(cid))
            heads.append(0)
    return boxes, scores, class_ids, heads, elapsed

#nms function
def nms(boxes, scores, conf_thr=0.5, nms_thr=0.4):
    """Return the indices of the boxes kept by OpenCV's non-maximum suppression."""
    if len(boxes) == 0:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf_thr, nms_thr)
    return np.array(keep).flatten().tolist()

#draw function (for now not in use)
def draw(img, boxes, scores, class_ids, keep=None):
    out = img.copy()
    idx = range(len(boxes)) if keep is None else keep
    for i in idx:
        x, y, w, h = boxes[i]
        c = colours[class_ids[i]]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        cv2.putText(out, f"{classes[class_ids[i]]} {scores[i]:.2f}", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return out

#get classes is a function to obtain all labels for each image
def getClasses(img_path):
    """Class id present in the label file for this image"""
    lbl = labels / (img_path.stem + ".txt")
    if not lbl.exists():
        return set()
    return{int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}

#creating th etest set to apply to images
tests = {
    "original": lambda im: im,
    "rotated":  lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
    "dark":     lambda im:(im * 0.25).astype("uint8"),
    "blurred":  lambda im:cv2.GaussianBlur(im, (31, 31), 0),
    "tiny":     lambda im:cv2.resize(im, None, fx=0.15, fy=0.15),
}

#loading the images
img_path= sorted(p for p in Img.iterdir() if p.suffix.lower() in{".jpg", ".jpeg", ".png"})
print(len(img_path), "Total images")
stats = {n: {"tp":0,"fp":0,"fn":0,"time":0.0, "conf":[], "with_det":0, "n":0 } for n in tests}
#testing for 5 images
for p in enumerate(img_path,1):
    img= cv2.imread(str(p))
    if img is None:
        continue
    truth = getClasses(p)


    for name, newImg in tests.items():
         im= newImg(img)
         boxes, scores, ids, heads, t = detect(net, im)
         keep = nms(boxes, scores)
         pred= {ids[i] for i in keep}

         #designing a stats for the images
         s= stats[name]
         s["n"] +=1
         s["time"]+= t
         s["conf"] += [scores[i] for i in keep]
         s["with_det"] +=1 if keep else 0
         s["tp"] += len(pred & truth)
         s["fp"] += len(pred - truth)
         s["fn"] += len(truth - pred)

         #getting an issue with the ids array
         found = [f"{classes[ids[i]]} {scores[i]:.2f}" for i in keep]
         print(f"{name:9s}: {found}      ({t*1000:.0f} ms)")

print(f"\nYolo11n     ({len(img_path)}images)")
print(f"{'test'}    {'precision'}    {'recall'}    {'avg conf'}    {'detected'}    {'ms/img'}")
for name, s in stats.items():
    prec= s["tp"]/ max(s["tp"] + s["fp"],1)
    rec = s["tp"]/ max(s["tp"] + s["fn"],1)
    conf = np.mean(s["conf"]) if s["conf"] else 0
    det = 100 * s["with_det"]/ max(s["n"],1)
    ms =  1000 * s["time"]/ max(s["n"],1)
    print(f"{name}   {prec}   {rec}   {conf}   {det}   {ms}")