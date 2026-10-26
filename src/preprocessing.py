# Merge datasets into one and move to "dataset/processed" folder for model training
# Check that images and labels are correct
# Remove duplicate and near-duplicate images
# Remap class IDs in the label files
# Retrieve valid image-label pairs and split files

from pathlib import Path
import random
import shutil
import numpy as np
from PIL import Image, ImageOps
import imagehash
from ultralytics.utils.ops import segments2boxes


ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT/"dataset"/"raw"
PROCESSED_DATA_DIR = ROOT/"dataset"/"processed"

TOTAL_ITEM_IMAGES = 500
TRAIN = 0.7
VAL = 0.2
TEST = 0.1

RANDOM_SEED = 42

CLASS_NAMES={
    0: "bottle",
    1: "keyboard",
    2: "laptop",
    3: "pendrive",
    4: "phone",
    5: "mug",
    6: "bin",
    7: "mouse"
}

SOURCE_CLASS_ID_MAPPING={
    "bottle-detection-1" :{ 0: 0},
    "Office-Keyboard-Detection-3": {0:1},
    "Laptop-1": {1:2},
    "Pendrive-detection-1": {0:3},
    "Pendrive-detection-6": {0:3},
    "Flash-drive-image-2": {0:3},
    "Phone-1": {0: 4},
    "Mug-OpenImages--1": {0:5},
    "Trash can.v1i.yolov8": {0:6},
    "Office-Mouse-1":{ 0:7}
}

DUPLICATE_THRESH = 5

SPLIT_PRIORITY = {"train": 0, "valid": 1, "test": 2, None:0}

def get_hashes(imgPath):
    ''' hash of image and its mirrored version '''
    with Image.open(imgPath) as source:
        img = ImageOps.exif_transpose(source).convert("RGB")

    h = imagehash.phash(img).hash.flatten()
    hm = imagehash.phash(ImageOps.mirror(img)).hash.flatten()
    return h, hm

def min_distance(h, hm, matrix):
    ''' smallest hamming distance between image or its mirror and each row of matrix '''
    d1 = (matrix != h).sum(axis=1)
    d2 = (matrix != hm).sum(axis=1)
    return np.minimum(d1, d2)

def remove_duplicates(pairs):
    ''' drop exact and near exact copies, keep first one only '''
    preserved = []
    matrix = np.zeros((len(pairs), 64), dtype=bool)
    count = 0

    for pair in pairs:
        try:
            h, hm = get_hashes(pair["image"])
        except Exception:
            continue

        if count and (min_distance(h, hm, matrix[:count]) <= DUPLICATE_THRESH).any():
            continue

        pair["hash"] = h
        pair["hash_mirror"] = hm

        matrix[count] = h
        count += 1
        preserved.append(pair)

    print(f"Duplicates/unreadable images removed: {len(pairs) - len(preserved)}")
    return preserved


def find_image_label_pairs():
    ''' get images with their matching labels '''
    image_extensions = {".jpg", ".png", ".jpeg"}
    valid_pairs = []
    missing_count = 0
    invalid_count = 0

    for sourceDataset, mapping in SOURCE_CLASS_ID_MAPPING.items():
        sourceDir = RAW_DATA_DIR / sourceDataset
        if not sourceDir.is_dir():
            print(f"{sourceDir} not found")
            continue

        for originalSplit in ("train", "valid", "test", None):
            base = sourceDir / originalSplit if originalSplit else sourceDir

            if not (base/"images").is_dir():
                continue

            for imagePath in sorted((base/"images").glob("*")):
                if imagePath.suffix.lower() not in image_extensions:
                    continue

                # path of corresponding label
                labelPath = base / "labels" / f"{imagePath.stem}.txt"

                if not labelPath.exists():
                    missing_count += 1
                    continue

                if validate_label(labelPath, mapping):
                    valid_pairs.append({
                        "image": imagePath,
                        "label": labelPath,
                        "source_dataset": sourceDataset,
                        "original_split": originalSplit
                    })

                else:
                    invalid_count += 1

    print(f"Number of images with matching labels file: {len(valid_pairs)}")
    print(f"Number of images missing labels file: {missing_count}")
    print(f"Number of images with incorrect labels: {invalid_count}")
    return valid_pairs


def validate_label(labelPath, classRemappedTo):
    ''' check if annotation file is correct '''
    try:
        with open(labelPath, "r", encoding="utf-8") as fileObj:
            content = fileObj.readlines()
            
    except Exception:
        return False

    # check if its empty
    if not any(line.strip() for line in content):
        return False

    found = False
    for line in content:
        if not line.strip():
            continue

        values = line.split()

        # check that the row contains 5 values only 
        if len(values) != 5:
            return False

        try:
            class_id = float(values[0])
            x_center, y_center, width, height = map(float, values[1:])

        except ValueError:
            return False

        if (not class_id.is_integer() or class_id < 0):
            return False

        # ignore unwanted classes
        if (class_id not in classRemappedTo):
            continue


        if not( 
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 <= width <= 1
            and 0 <= height <= 1
        ):
            return False
        
        found = True

    return found

def save_remapped_label(sourcePath, outputPath, classMapping):
    ''' copy label file with its remapped id '''
    newLines = []

    with open(sourcePath, "r", encoding="utf-8") as fileObj:
        for line in fileObj:
            if not line.strip():
                continue

            values = line.split()
            source_id = float(values[0])

            # drop unwanted classes
            if source_id not in classMapping:
                continue
            
            # change class id to match defined one
            values[0] = str(classMapping[source_id])
            newLines.append(" ".join(values))

    # write modified annotation lines
    with open(outputPath, "w", encoding="utf-8") as fileObj:
        fileObj.write("\n".join(newLines) + "\n")


def extract_images():
    ''' select images for dataset '''
    validPairs = find_image_label_pairs()

    validPairs = remove_duplicates(validPairs)

    for pair in validPairs:
        labelPath = pair["label"]
        classMapping = SOURCE_CLASS_ID_MAPPING[pair["source_dataset"]]

        targetClasses = set()

        with open(labelPath, "r", encoding="utf-8") as fileObj:
            for line in fileObj:
                if line.strip():
                    sourceID = int(float(line.split()[0]))
                    if (sourceID in classMapping):
                        targetClasses.add(classMapping[sourceID])

        pair["target_classes"] = targetClasses
    validPairs= [pair for pair in validPairs if pair["target_classes"]]
    print(f"Valid image-label pairs collected: {len(validPairs)}")

    return validPairs

def split_dataset():
    """Split images into train, valid and test folders."""

    selectedPairs = extract_images()

    if not selectedPairs:
        print("No valid images found.")
        return

    # Group images by source dataset.
    datasets = {}

    for pair in selectedPairs:
        sourceDataset = pair["source_dataset"]
        datasets.setdefault(sourceDataset, []).append(pair)

    candidateSplits = {
        "train": [],
        "valid": [],
        "test": []
    }

    randomRange = random.Random(RANDOM_SEED)

    for sourceDataset, pairs in datasets.items():

        # Separate images by their original split
        originalSplitPairs = {
            splitName: [
                pair for pair in pairs
                if pair["original_split"] == splitName
            ]
            for splitName in ("train", "valid", "test")
        }
        
        # unsplit datasets
        unsplitPairs = [
            pair for pair in pairs
            if pair["original_split"] is None
        ]

        # Preserve original splits
        for splitName in ("train", "valid", "test"):
            candidateSplits[splitName].extend(
                originalSplitPairs[splitName]
            )

        # Create new 70/20/10 splits for datasets that are not split
        randomRange.shuffle(unsplitPairs)
        totalImages = len(unsplitPairs)
        trainEnd = int(totalImages * TRAIN)
        validEnd = trainEnd + int(totalImages * VAL)

        candidateSplits["train"].extend(
            unsplitPairs[:trainEnd]
        )
        candidateSplits["valid"].extend(
            unsplitPairs[trainEnd:validEnd]
        )
        candidateSplits["test"].extend(
            unsplitPairs[validEnd:]
        )

    quotas = {
        "train": int(TOTAL_ITEM_IMAGES * TRAIN),
        "valid": int(TOTAL_ITEM_IMAGES * VAL),
        "test": int(TOTAL_ITEM_IMAGES * TEST)
    }

    splitPairs = {
        "train": [],
        "valid": [],
        "test": []
    }

    usedImages = set()

    for splitName, quota in quotas.items():

        classCounts = {
            classID: 0 for classID in CLASS_NAMES
        }

        candidates = candidateSplits[splitName].copy()
        randomRange.shuffle(candidates)

        # Select images until each class reaches its quota or no more eligible images are available
        for classID in CLASS_NAMES:

            for pair in candidates:
                if classCounts[classID] >= quota:
                    break

                imageKey = str(pair["image"].resolve())

                if imageKey in usedImages:
                    continue

                targetClasses = pair["target_classes"]

                if classID not in targetClasses:
                    continue

                # Avoid exceeding the quota for any class
                if any(
                    classCounts[c] >= quota
                    for c in targetClasses
                ):
                    continue

                splitPairs[splitName].append(pair)
                usedImages.add(imageKey)

                for c in targetClasses:
                    classCounts[c] += 1

                candidates.remove(pair)

        print(f"\n{splitName.upper()} selection:")
        for classID, className in CLASS_NAMES.items():
            print(
                f"  {className}: "
                f"{classCounts[classID]}/{quota}"
            )

    # Create the processed folder
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    # Create subfolders
    for splitName in ("train", "valid", "test"):
        (PROCESSED_DATA_DIR / splitName / "images").mkdir(
            parents=True, exist_ok=True
        )
        (PROCESSED_DATA_DIR / splitName / "labels").mkdir(
            parents=True, exist_ok=True
        )

    # Rename, copy and remap images and labels.
    classFileCounts = {classID: 0 for classID in CLASS_NAMES}

    splitCounts = {
        splitName: {classID: 0 for classID in CLASS_NAMES}
        for splitName in splitPairs
    }

    for splitName, pairs in splitPairs.items():
        for pair in pairs:
            imagePath = pair["image"]
            labelPath = pair["label"]

            primaryClass = min(pair["target_classes"])
            classFileCounts[primaryClass] += 1

            fileNumber = classFileCounts[primaryClass]
            # rename files
            newStem = (f"{CLASS_NAMES[primaryClass]}_{fileNumber:03d}")

            outputImage = (
                PROCESSED_DATA_DIR
                / splitName
                / "images"
                / f"{newStem}{imagePath.suffix.lower()}"
            )

            shutil.copy2(imagePath, outputImage)

            outputLabel = (
                PROCESSED_DATA_DIR
                / splitName
                / "labels"
                / f"{newStem}.txt"
            )

            save_remapped_label(
                labelPath,
                outputLabel,
                SOURCE_CLASS_ID_MAPPING[pair["source_dataset"]]
            )

            # count images in each class
            for classID in pair["target_classes"]:
                splitCounts[splitName][classID] += 1

    print("\nProcessed images per split:")

    for splitName, counts in splitCounts.items():
        print(f"\n{splitName}: {sum(counts.values())} class occurrences")

        for classID, className in CLASS_NAMES.items():
            print(
                f"  {className}: {counts[classID]}"
            )

if __name__ == "__main__":
    split_dataset()