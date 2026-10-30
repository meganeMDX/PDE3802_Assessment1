from pathlib import Path
import shutil
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT / "dataset" / "raw"

STAPLE_DIR = RAW_DATA_DIR / "stapler"
IMAGE_DIR = RAW_DATA_DIR / "stapler_arranged"/ "images"
LABEL_DIR = RAW_DATA_DIR / "stapler_arranged"/ "labels"

CLASS_ID = 1

def move_folder():
    ''' move copies to new folder and seperate images and labels '''
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    imageCount = 0
    labelCount = 0

    for file in STAPLE_DIR.iterdir():
        if file.is_file():
            if file.suffix.lower() == ".jpg":
                shutil.copy(file, IMAGE_DIR/file.name)
                imageCount += 1

            elif file.suffix.lower() == ".txt":
                shutil.copy(file, LABEL_DIR/file.name)
                labelCount += 1

    print("Number of images copied: ", imageCount)
    print("Number of labels copied: ", labelCount)
    

def convert_annotations():
    ''' convert annotation file from pixel format to yolo format '''
    convertedCount = 0
    invalidCount = 0
    for labelFile in LABEL_DIR.glob("*.txt"):
        imgPath = IMAGE_DIR / f"{labelFile.stem}.jpg"

        if not imgPath.exists():
            print("Missing image for: ", labelFile.name)
            invalidCount += 1
            continue

        # image dimensions 
        with Image.open(imgPath) as img:
            width, height = img.size

        # read annotations
        with open(labelFile, "r", encoding="utf-8") as fileObj:
            lines = fileObj.readlines()

        labels = []

        for line in lines:
            values = line.split()
            if not values:
                continue

            if len(values) != 4:
                print(f"Invalid annotation in {labelFile.name}")
                invalidCount += 1
                continue

            x1, y1, x2, y2 = map(float, values)
            x_min= min(x1, x2)
            x_max= max(x2, x1)

            y_min= min(y1, y2)
            y_max= max(y2, y1)

            # convert to yolo format
            if not ( 0 <= x_min < x_max <= width and 0<= y_min < y_max <= height):
                print(f"Invalid annotation in {labelFile.name}")
                invalidCount += 1
                continue

            x_center = (x_min + x_max) / (2 * width)
            y_center = (y_min + y_max) / (2 * height)

            box_width = (x_max - x_min) / width
            box_height = (y_max - y_min) / height


            labels.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        if len(labels) > 0:
            with open(labelFile, "w", encoding="utf-8") as file:
                file.write("\n".join(labels) + "\n")
            convertedCount += 1

    print("Number of converted annotations: ", convertedCount)
    print("Number of invalid annotations: ", invalidCount)


if __name__ == "__main__":
    move_folder()
    convert_annotations()