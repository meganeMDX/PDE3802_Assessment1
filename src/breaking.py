import cv2, numpy as np, time

#recreate functions from yolo_utils from lab
CLASSES = open("data.names").read().strip().split("\n")
COLOURS = np.random.default_rng(42).integers(0,255, size=(len(CLASSES),3)).tolist()

#loading the data for yolo (cfg and weights)
def load_net(cfg="models/yolov3.cfg", weights="models/yolov3.weights"):
    return cv2.dnn.readNetFromDarknet(cfg, weights)

#function to detect the object
def detect(net, img, size=416, conf_thr=0.5):
    """Run YOLO and return every box above conf_thr (BEFORE NMS)."""
    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    elapsed = time.time() - start
 
    boxes, scores, class_ids, heads = [], [], [], []
    for head_index, out in enumerate(outputs):          # 0 = coarse grid ... last = finest grid
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
 
def nms(boxes, scores, conf_thr=0.5, nms_thr=0.4):
    """Return the indices of the boxes kept by OpenCV's non-maximum suppression."""
    if len(boxes) == 0:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf_thr, nms_thr)
    return np.array(keep).flatten().tolist()
 
def draw(img, boxes, scores, class_ids, keep=None):
    out = img.copy()
    idx = range(len(boxes)) if keep is None else keep
    for i in idx:
        x, y, w, h = boxes[i]
        c = COLOURS[class_ids[i]]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        cv2.putText(out, f"{CLASSES[class_ids[i]]} {scores[i]:.2f}", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return out

#main code from breaking.py from lab
net = load_net()
img = cv2.imread("images/fromPhone.jpeg")

tests= {
    "original": img,
    "rotated":  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    "dark":     (img * 0.25).astype("uint8"),
    "blurred":  cv2.GaussianBlur(img, (31, 31), 0),
    "tiny":     cv2.resize(img, None, fx=0.15, fy=0.15),
}

for name, im in tests.items():
    boxes, scores, ids, heads, t = detect(net, im)
    keep = nms(boxes, scores)
    found = [f"{CLASSES[ids[i]]} {scores[i]:.2f}" for i in keep]
    print(f"{name:9s}: {found}")