import cv2, numpy as np, time,yaml
from pathlib import Path
from ultralytics import YOLO

#creating dedicated paths for access to each required folder
root = Path(__file__).resolve().parent
dataYAML = root/"dataset"/"data.yaml"
Img =  root/"dataset"/ "processed"/ "test" /"images"
labels = root/"dataset"/ "processed"/ "test" /"labels"
weights = root / "weights"/ "yolo26n.pt"

#creating the classes from data.yaml
with open(dataYAML) as file:
    names = yaml.safe_load(file)["names"]
classes = list(names.values())

# either saving the classes as a dictionary with each class enumerated or create a list
#depending on the version used
if isinstance:
    (names, dict)
else: list(names)

#colours
colours= np.random.default_rng(42).integers(0, 255, size=(len(classes), 3)).tolist()

#function to load the configuration for yolov8
def load_net(weights=str(weights)):
    return YOLO(weights)

#running thresholding before and nnms is applied
def detect(net, img, size =416, conf_thr=0.5):
    """Run YOLO and return every box above conf_thr (BEFORE NMS)."""
    start = time.time()
    H, W = img.shape[:2]
    outputs = net.predict(img, img_size=size,conf=conf_thr, iou=1.0, verbose=False)[0]
    elapsed = time.time() - start

    boxes, scores, class_ids, heads = [], [], [], []
    for head_index, out in enumerate(outputs):
            for row in out:
                class_scores = row[5:]
                cid = int(np.argmax(class_scores))
                score = float(class_scores[cid])
                if score > conf_thr:
                    cx, cy, w, h = row[:4] * np.array([W, H, W, H])
                    boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                    scores.append(score)
                    class_ids.append(cid)
                    heads.append(head_index)
    return boxes, scores, class_ids, heads, elapsed

#nms function
def nms(boxes, scores, conf_thr=0.5, nms_thr=0.4):
    """Return the indices of the boxes kept by OpenCV's non-maximum suppression."""
    if len(boxes) == 0:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf_thr, nms_thr)
    return np.array(keep).flatten().tolist()

#draw function
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

#calling the above functions to test and break system
#calling load_net to ipplement weights 
net= load_net()

#get classes is a function to obtain all labels for each image
def getClasses(img_path):
    """Class id present in the label file for this image"""
    lbl = labels / (img_path.stem + ".txt")
    if not lbl.exists():
        return set()
    return{int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
    
#loading the images
img_path= sorted(p for p in Img.iterdir() if p.suffix.lower in{".jpg", ".jpeg", ".png"})
tests = {
    "original": lambda im: im,
    "rotated":  lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
    "dark":     lambda im:(im * 0.25).astype("uint8"),
    "blurred":  lambda im:cv2.GaussianBlur(im, (31, 31), 0),
    "tiny":     lambda im:cv2.resize(im, None, fx=0.15, fy=0.15),
}
for p in enumerate(img_path):
    img= cv2.imread(str(p))
    if img is None:
        continue
    truth = getClasses(p)


    for name, newImg in tests.items():
         im= newImg(img)
         boxes, scores, ids, heads, t = detect(net, im)
         keep = nms(boxes, scores)
         found = [f"{classes[ids[i]]} {scores[i]:.2f}" for i in keep]
         print(f"{name:9s}: {found}")