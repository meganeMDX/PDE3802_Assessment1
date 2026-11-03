from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "dataset" / "processed" / "train"
VALID_DIR = ROOT / "dataset" / "processed" / "valid"
TEST_DIR = ROOT / "dataset" / "processed" / "test"

CLASS_IDS={
    "bottle": 0,
    "scissors": 1,
    "laptop": 2,
    "smartphone": 3,
    "mug": 4,
    "bin": 5,
    "mouse": 6
}

IMAGE_EXT = {".jpg", ".png", ".jpeg"}

def check_img_format(folder):
    ''' check all images are in correct format '''
    count_invalid_format = 0

    for image_path in (folder/"images").glob("*"):
        if image_path.suffix.lower() not in IMAGE_EXT:
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
    incorrect_count = 0

    for label_path in (folder/"labels").glob("*"):
        label_name = label_path.stem
        expected_class_id = CLASS_IDS.get(label_name.split("_")[0])
        if expected_class_id is None:
            print("Invalid annotation")
            incorrect_count += 1
            break
        
        try:
            with open(label_path, "r", encoding="utf-8") as file_obj:
                content = file_obj.readlines()

            if not any(line.strip() for line in content):
                incorrect_count += 1
                continue

            for line in content:
                if not line.strip():
                    continue

                if (not check_annotation_line(line, expected_class_id)):
                    incorrect_count += 1

        except Exception as e:
            print("Error openeing label file: ", e)

    return incorrect_count

def sets_count(folder):
    ''' get number of images and labels '''
    image_count = 0
    label_count = 0

    for file in (folder/"images").glob("*"):
        if file.suffix.lower() in IMAGE_EXT:
            image_count += 1

    for file in (folder/"labels").glob("*.txt"):
        label_count += 1

    return (image_count, label_count)

def image_count_per_class(folder):
    ''' get number of images per class '''
    counts = {name: 0 for name in CLASS_IDS}

    for file in (folder/"images").glob("*"):
        if file.suffix.lower() in IMAGE_EXT:
            label_name = file.stem
            class_name = label_name.split("_")[0]
            counts[class_name] += 1

    return counts

def plot_image_distribution(classCounts, title):
    classes = list(classCounts.keys())
    counts = list(classCounts.values())

    plt.figure(figsize=(10,6))
    bars = plt.bar(classes, counts, color="mediumpurple")
    plt.bar_label(bars, padding=3)
    plt.title(title)
    plt.xlabel('Object Class', fontweight='bold')
    plt.ylabel('Number of Images', fontweight='bold')
    plt.show()


def run_checks(folder, name):
    image_count, label_count = sets_count(folder)    
    no_labels, no_images = check_pairs(folder)
    class_counts = image_count_per_class(folder)

    print(f"{name} set")
    print(f"Number of Images: {image_count}")
    print(f"Number of Labels: {label_count}")
    
    print(f"\nImages per class")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    print("\nChecks")
    print(f"Incorrect Image Format: {check_img_format(folder)}")
    print(f"Images without labels: {len(no_labels)}")
    print(f"Labels without images: {len(no_images)}")
    print(f"Incorrect annotations: {check_annotations(folder)}\n\n")

    plot_image_distribution(class_counts, f"Image Distribution in {name} folder")


if __name__ == "__main__":
    run_checks(TRAIN_DIR, "Training")
    run_checks(VALID_DIR, "Validation")
    run_checks(TEST_DIR, "Testing")