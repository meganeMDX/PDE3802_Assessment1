import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import shutil
import yaml

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR =  ROOT / "dataset" / "processed" / "test" / "images"
LABEL_DIR = ROOT / "dataset" / "processed" / "test" / "labels"
MODEL_PATH = ROOT / "model" / "best.pt"
SAVE_DIR = ROOT / "filter_test_preview"
FILTERED_DIR = ROOT / "filter_dataset"

IMAGES_NUM = 5

net = YOLO(str(MODEL_PATH))
print("model load net:", net.names)
classes = [net.names[k] for k in sorted(net.names)]
colours = np.random.default_rng(42).integers(0, 255, size=(len(classes), 3)).tolist()

#filters to apply to images
tests = {
    "original": lambda img: img,
    "rotated":  lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    "dark":     lambda img:(img * 0.25).astype("uint8"),
    "blurred":  lambda img:cv2.GaussianBlur(img, (31, 31), 0),
    "tiny":     lambda img:cv2.resize(img, None, fx=0.15, fy=0.15),
}

def save(filter, img_path, img, boxes, scores, ids):
    ''' save image with filter applied to it and the bounding boxes '''
    output_path = SAVE_DIR / f"{img_path.stem}_{filter}.jpg"
    cv2.imwrite(str(output_path), draw(img, boxes, scores, ids))

def rotate_label(annotations):
    ''' rotate yolo label 90 degrees clockwise for rotated filter '''
    output =[]
    for line in annotations.splitlines():
        if line.strip():
            c, x, y, w, h = line.split()
            x, y, w, h= map(float, (x, y, w, h))
            output.append(f"{c} {1-y:.6f} {x:.6f} {h:.6f} {w:.6f}")
    return "\n".join(output) + "\n"


def build_set(name, fn, image_paths):
    ''' write dataset for filtered images and yaml '''
    root = FILTERED_DIR/name
    shutil.rmtree(root, ignore_errors=True)
    (root/"images").mkdir(parents = True)
    (root/"labels").mkdir(parents = True)

    for p in image_paths:
        img= cv2.imread(str(p))

        if img is None:
            continue

        cv2.imwrite(str(root / "images" / f"{p.stem}.png"), fn(img))
        label = (LABEL_DIR/f"{p.stem}.txt").read_text()
        if name=="rotated":
            label= rotate_label(label)

        (root/"labels"/f"{p.stem}.txt").write_text(label)

    yml = root/"data.yaml"
    yml.write_text(yaml.safe_dump({"path": str(root.resolve()),
                                    "val": "images",
                                    "train": "images",
                                    "names": dict(net.names)
                                }))

    return yml

def get_map_scores(name, fn, image_paths):
    ''' evaluate model on the images to get mAP scores '''
    yml = build_set(name, fn, image_paths)
    results = net.val(data = str(yml), imgsz = 640, verbose=False, plots=False)
    return results.box.map, results.box.map50, results.box.mp, results.box.mr


def detect(net, img):
    """apply filters and run detection model"""

    outputs = net.predict(img, imgsz=640, verbose=False)[0]

    h, w = img.shape[:2]
    
    boxes, scores, class_ids = [], [], []

    for row, score, cid in zip(outputs.boxes.xywhn.tolist(), outputs.boxes.conf.tolist(),outputs.boxes.cls.tolist()):
        cx, cy, bw, bh = np.array(row) * np.array([w,h,w,h])
        boxes.append([int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)])
        scores.append(float(score))
        class_ids.append(int(cid))

    return boxes, scores, class_ids


def draw(img, boxes, scores, class_ids, keep=None):
    ''' draw predicted bounding boxes on image '''
    out = img.copy()

    idx = range(len(boxes)) if keep is None else keep
    for i in idx:
        x, y, w, h = boxes[i]
        c = colours[class_ids[i]]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        cv2.putText(out, f"{classes[class_ids[i]]} {scores[i]:.2f}", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return out

def getClasses(img_path):
    """Class id present in the label file for this image"""
    lbl = LABEL_DIR / (img_path.stem + ".txt")
    if not lbl.exists():
        raise FileNotFoundError(f"Missing label file: {lbl}")
    
    return{int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}


def select_images(img_path):
    ''' retrieves 5 images per class item in test set '''
    selected_images = set()
    rng = np.random.default_rng(42)

    for class_id in range(len(classes)):
        class_imgs = [p for p in img_path if class_id in getClasses(p)]
        rng.shuffle(class_imgs)
        chosen = class_imgs[:IMAGES_NUM]

        for p in chosen:
            selected_images.add(p)

    return sorted(selected_images)


def test():
    # load images
    test_img_path= sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in{".jpg", ".jpeg", ".png"})

    img_path = select_images(test_img_path)
    print(len(img_path), " images selected")

    stats = {n: {"with_det":0, "n":0 } for n in tests}

    if SAVE_DIR.exists():
        shutil.rmtree(SAVE_DIR, ignore_errors=True)

    SAVE_DIR.mkdir(parents=True)

    # predictions
    for p in img_path:
        img= cv2.imread(str(p))

        if img is None:
            continue

        for name, newImg in tests.items():
            im = newImg(img)
            boxes, scores, ids= detect(net, im)

            # update stats
            s = stats[name]
            s["n"] +=1
            s["with_det"] +=1 if ids else 0
            save(name, p, im, boxes, scores, ids)

    # get mAP scores
    mAP = {name: get_map_scores(name, fn, img_path) for name, fn in tests.items()}
    shutil.rmtree(FILTERED_DIR, ignore_errors=True)

    # final results
    print(f"\nYolo11n     ({len(img_path)}images)")
    print(f"{'Test':<12}   {'mAP50-95':<10}    {'mAP50':<10}    {'Mean Precision':<20}    {'Mean Recall':<20}   {'Det Rate (%)':<12}")
    for name, s in stats.items():
        mAP50_95, mAP50, mp, mr = mAP[name]
        det = 100 * s["with_det"]/ max(s["n"],1)
        print(f"{name:<12}   {mAP50_95:<10.3f}    {mAP50:<10.3f}    {mp:<20.3f}    {mr:<20.3f}   {det:<12.2f}")

if __name__ == "__main__":
    test()