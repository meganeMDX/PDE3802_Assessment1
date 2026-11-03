import cv2
import random
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "dataset"/ "processed" / "train" / "images"
LABELS_DIR = ROOT / "dataset"/ "processed" / "train" / "labels"
SAVE_DIR = ROOT / "preview_with_bounding_boxes"

SAMPLE_NUM = 10

CLASS_NAMES={
  0: "bottle",
  1: "scissors",
  2: "laptop",
  3: "smartphone",
  4: "mug",
  5: "bin",
  6: "mouse"
}

random.seed(42)

def create_preview_images(selectedImages):
  for img in selectedImages:
      # get image and label path
      img_path = Path(IMAGES_DIR / img)
      label_path = Path(LABELS_DIR / (Path(img).stem + ".txt"))

      preview = cv2.imread(img_path)
      height, width = preview.shape[:2]

      with open(label_path, "r", encoding="utf-8") as annotation:
          lines = annotation.readlines()
          for line in lines:
            class_id = int(line[0])
            x, y, w, h = map(float, line.split()[1:])

            # add bounding box to image
            left = max(int((x - w/2) * width), 0)
            right = min(int((x + w/2) * width), width-1)
            top = max(int((y - h/2) * height), 0)
            bottom = min(int((y + h/2) * height), height-1)

            label_y = top - 3
            if label_y < 20:
              label_y = top + 20

            cv2.rectangle(preview, (left, top), (right, bottom), (0, 0, 0), 2)
            cv2.putText(preview, CLASS_NAMES[class_id], ( left + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imwrite(SAVE_DIR/img, preview)

if __name__ == "__main__":

  if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR, ignore_errors=True)
  
  SAVE_DIR.mkdir(parents=True)

  image_files = os.listdir(IMAGES_DIR)

  for item in CLASS_NAMES.values():
    item_images = [img for img in image_files if img.startswith(item) ]
    selected_images = random.sample(item_images, SAMPLE_NUM)
    create_preview_images(selected_images)
    print(f"annoted previews for {item} saved")