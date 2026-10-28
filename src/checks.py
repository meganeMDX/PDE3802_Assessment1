from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT/"dataset"/"processed"/"train"
VALID_DIR = ROOT/"dataset"/"processed"/"valid"
TEST_DIR = ROOT/"dataset"/"processed"/"test"

CLASS_IDS={
    "bottle": 0,
    "keyboard": 1,
    "laptop": 2,
    "pendrive": 3,
    "phone": 4,
    "mug": 5,
    "bin": 6,
    "mouse": 7
}

IMAGE_EXT = {".jpg", ".png", ".jpeg"}

def check_img_format(folder):
    ''' check all images are in correct format '''
    count_invalid_format = 0

    for imagePath in (folder/"images").glob("*"):
        if imagePath.suffix.lower() not in IMAGE_EXT:
            count_invalid_format += 1

    return count_invalid_format

def check_pairs(folder):
    ''' return images with no corresponding labels and labels with no corresponding images '''
    images = {path.stem for path in (folder/"images").glob("*")}
    labels = {path.stem for path in (folder/"labels").glob("*.txt")}
    return (sorted(images - labels), sorted(labels - images))

def check_annotation_line(line, expectedClassID):
    ''' check annotation is valid and has correct class ID '''
    values = line.split()

    if len(values) != 5:
        return False
    try:
        class_id = float(values[0])
        x_center, y_center, width, height = map(float, values[1:])

    except ValueError:
        return False

    # check class id
    if (not class_id.is_integer() or class_id != expectedClassID):
        return False

    # check coordinates and dimensions
    if not( 
        0 <= x_center <= 1
        and 0 <= y_center <= 1
        and 0 <= width <= 1
        and 0 <= height <= 1
    ):
        return False
    
    return True


def check_annotations(folder):
    ''' check label annotations '''
    incorrectCount = 0

    for labelPath in (folder/"labels").glob("*"):
        labelName = labelPath.stem
        expectedClassID = CLASS_IDS.get(labelName.split("_")[0])
        if expectedClassID is None:
            print("Invalid annotation")
            incorrectCount += 1
            break
        
        try:
            with open(labelPath, "r", encoding="utf-8") as fileObj:
                content = fileObj.readlines()

            if not any(line.strip() for line in content):
                incorrectCount += 1
                continue

            for line in content:
                if not line.strip():
                    continue

                if (not check_annotation_line(line, expectedClassID)):
                    incorrectCount += 1

        except Exception as e:
            print("Error openeing label file: ", e)

    return incorrectCount

def sets_count(folder):
    ''' get number of images and labels '''
    imageCount = 0
    labelCount = 0

    for file in (folder/"images").glob("*"):
        if file.suffix.lower() in IMAGE_EXT:
            imageCount += 1

    for file in (folder/"labels").glob("*.txt"):
        labelCount += 1

    return (imageCount, labelCount)

def image_count_per_class(folder):
    ''' get number of images per class '''
    counts = {name: 0 for name in CLASS_IDS}

    for file in (folder/"images").glob("*"):
        if file.suffix.lower() in IMAGE_EXT:
            labelName = file.stem
            class_name = labelName.split("_")[0]
            counts[class_name] += 1

    return counts

def plot_image_distribution(classCounts, title):
    classes = list(classCounts.keys())
    counts = list(classCounts.values())

    plt.figure(figsize=(10,6))
    plt.bar(classes, counts, color="mediumpurple")
    plt.title(title)
    plt.xlabel('Object Class', fontweight='bold')
    plt.ylabel('Number of Images', fontweight='bold')
    plt.show()


def run_checks(folder, name):
    imageCount, labelCount = sets_count(folder)    
    noLabels, noImages = check_pairs(folder)
    classCounts = image_count_per_class(folder)

    print(f"{name} set")
    print(f"Number of Images: {imageCount}")
    print(f"Number of Labels: {labelCount}")
    
    print(f"\nImages per class")
    for className, count in classCounts.items():
        print(f"{className}: {count}")

    print("\nChecks")
    print(f"Incorrect Image Format: {check_img_format(folder)}")
    print(f"Images without labels: {len(noLabels)}")
    print(f"Labels without images: {len(noImages)}")
    print(f"Incorrect annotations: {check_annotations(folder)}\n\n")

    plot_image_distribution(classCounts, f"Image Distribution in {name} folder")


if __name__ == "__main__":
    run_checks(TRAIN_DIR, "Training")
    run_checks(VALID_DIR, "Validation")
    run_checks(TEST_DIR, "Testing")