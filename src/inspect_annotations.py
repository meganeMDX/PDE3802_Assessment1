import cv2
import random
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT/"dataset"/"processed"/"train"/"images"
LABELS_DIR = ROOT/"dataset"/"processed"/"train"/"labels"
SAVE_DIR = ROOT/"preview_with_bounding_boxes"

SAMPLE_NUM = 10

classNames = {
  0: "bottle",
  1: "keyboard",
  2: "laptop",
  3: "pendrive",
  4: "phone",
  5: "mug",
  6: "bin",
  7: "mouse",
}

random.seed(42)
imagefiles = os.listdir(IMAGES_DIR)

def create_preview_images(selectedImages):
  for img in selectedImages:
      # get image and label path
      imgPath = Path(IMAGES_DIR / img)
      labelPath = Path(LABELS_DIR / (Path(img).stem + ".txt"))

      preview = cv2.imread(imgPath)
      height, width = preview.shape[:2]

      with open(labelPath, "r", encoding="utf-8") as annotation:
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
            cv2.putText(preview, classNames[class_id], ( left + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imwrite(SAVE_DIR/img, preview)

for item in classNames.values():
  itemImages = [img for img in imagefiles if img.startswith(item) ]
  selectedImages = random.sample(itemImages, SAMPLE_NUM)
  create_preview_images(selectedImages)
  print(f"annoted previews for {item} saved")