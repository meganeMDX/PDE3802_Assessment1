# Indoor Office Object Detection Coursework Project

## 1. Project Overview
This project focuses on developing a computer vision system that detects and identifies office items in images or live camera feed. It uses the **YOLO11n** object detection model from Ultralytics to locate and classify objects belonging to the following seven classes: 
- Bottle
- Scissors
- Laptop
- Smartphone
- Mug
- Bin
- Mouse

The system is developed with the aim of supporting an autonomous indoor office robot by enabling it to detect and locate these objects in its environment.

The project includes a Streamlit web application that allows users to upload images, capture images or use a live camera feed for object detection. The application then displays the detection results, including the predicted class, confidence score and  bounding box for each detected object.

## 2. Application Features
- **Image Upload:** Allow users to upload an image for object detection.
- **Live Camera:** Enables real-time object detection using the live camera feed.
- **Take picture from camera:** Allow users to capture a still image from the camera and run the object detection model on the captured image.
- **Object Detection:** Uses the trained YOLO11n model to detect and locate objects belonging to the seven predefined classes.
- **Detection Results:** Displays the predicted class, bounding box and confidence score for detected objects.
- **Model Performance Popup:** Displays the model performance metrics in a popup window.

## 3. Folder Structure

```text
.
├── .streamlit/
│   └── config.toml
│
├── assets/  
│
├── dataset/  
│   ├── own_images/
│   │   
│   ├── processed/
│   │   
│   ├── data.yaml
│   │   
│   └── DATASET_CARD.md
│
├── filter_test_preview/  
│
├── model/  
│
├── notebook/
│   └── model_training.ipynb
│
├── preview_with_bounding_boxes/
│
├── requirements.txt
│
├── runs/
│
├── src/
│   ├── breaking.py
│   │   
│   ├── checks.py
│   │   
│   ├── inspect_annotations.py
│   │   
│   ├── preprocessing.py
│   │   
│   ├── test.py
│   │   
│   ├── train.py
│   │   
│   └── tuning.py
│
├── ui/
│   ├── config.py
│   │   
│   └── styles.py
│
├── weights/
│
├── app.py
│
└── README.md
```


**Main Folder and Files**

| File/Folder    | Description |
| :-------- | :------- |
| .streamlit/config.toml | Configuration settings for the Streamlit application |
| assets/  |  Contains supporting images, including dataset class distribution graphs and UI screenshots. |
| dataset/  | Contains the processed dataset, the YOLO dataset configuration file (data.yaml) and the dataset card.    |
| filter_test_preview/  |  Contains images showing model's detection results after applying different image transformations to a subset of the test set   |
| model/  |  Contains a copy of the final model used by the interface for object detection.   |
| notebook/    |  Contains the Google Colab notebook used for model training, tuning and testing.   |
| preview_with_bounding_boxes/  |   Contains preview images from the training set with bounding boxes drawn around annotated objects.  |
| requirements.txt |  Contains the dependencies required to run the application.  |
| runs/ |  Stores outputs generated during model training and testing.   |
| src/    |  Contains python scripts for preprocessing, dataset checks, annotation inspection, model training, hyperparameter tuning and testing. |
| ui/  |   Contains configuration and styling files used by the Streamlit interface. |
| app.py    |  Entry point for the Streamlit interface.  |


**Scripts in src/**

| File    | Description |
| :-------- | :------- |
| breaking.py  |  Applies image transformation (rotation, brightness reduction, blur and size reduction ) to evaluate the model robustness under different image conditions. |
| checks.py  | Performs checks on the final merged dataset.    |
| inspect_annotations.py |  Generate previews of training images with their bounding boxes.   |
| preprocessing.py   |  Preprocesses, retrieves valid image-label pairs and combines final dataset.  |
| test.py    |  Tests the trained model and evaluates its performance.   |
| train.py    | Trains the YOLO object detection model (baseline training).  |
| tuning.py    |  Performs hyperparameter tuning to explore model configurations and improve performance.  |

Model training, tuning and testing were performed using Google Colab for GPU acceleration. The scripts are available in **src/** directory.

For details about the dataset, including its organisation, sources, licensing information and the rationale behind the dataset and model selection, see the [Dataset card](dataset/DATASET_CARD.md).

## 4. Requirements
Before running the application, ensure that the following are available:

- Python installed on your system.
- The project files, including the trained YOLO model weights located at `model/best.pt`.
- A webcam or camera device if you intend to use the live camera features.

## 5. Installation
### 5.1 Dependencies
The application uses the following Python libraries:
- `streamlit` : Provides the application interface.
- `Pillow`: Handles image processing.
- `ultralytics`: For the YOLO11n model.
- `streamlit-webrtc`: Enables live camera streaming through the Streamlit interface.
- `av`: Handles video frame processing.

### 5.2 Installation Steps
1. Clone the repository from GitHub:
```bash
git clone https://github.com/meganeMDX/PDE3802_Assessment1.git
``` 
2. Navigate to the project directory
```bash
cd PDE3802_Assessment1
```
3. Install the required dependencies by running the following in the terminal:
```bash
pip install -r requirements.txt
```

## 6. How to Run
Follow the steps below to run the application after installing the dependencies:
1. Open a terminal in the project's root directory
```
cd PDE3802_Assessment1
```

2. Run the following command in the terminal to start the application:
```
streamlit run app.py
```

3. The streamlit web application will open in your default browser. If it does not open automatically, follow the **local URL** displayed in the terminal.

4. To stop the application, return to the terminal where the Streamlit application is running and press `Ctrl + C`.

## 7. Using the Application
Once the application opens in your browser, you can use the following features:

### Upload an Image
Upload an image containing office items and click on **Analyse Image** button. The application displays the uploaded image with the bounding boxes drawn around detected objects, their predicted class labels and confidence scores.

[Insert annotated image of where to upload image and button to click]

[Insert annotated image of analysed result section]

### Live Camera
Click the **Live Camera** Button. Real-time detection is selected by default. Click **START** to begin real-time detection. Detected objects, their bounding boxes and confidence scores will appear on the video frame. 

You can click on **SELECT DEVICE** to choose a different camera or video input device.

[Insert annotated image of live detection - with button to click]

[Insert image of live detection example]

### Capture an Image from the Camera
Select **Take a Photo** to open the camera capture interface. Click **Take Photo** below the camera feed to capture an image. Then, click on **Analyse Image** to run object detection on the captured image.

[Insert annotated image of what button to click to capture image]

### View Model Performance Metrics
Click the information icon to open a popup displaying details about the model and its performance based on the held-out test data.

[Insert annotated image of what button to click to open popup]
[Insert image of popup]

## 8. Expected Outputs
The application displays object detection results based on the input image or camera feed.

The expected outputs include:
- **Detected Objects:** The predicted class labels of detected office items, including bottle, scissors, laptop, smartphone, mug, bin, and mouse.
- **Bounding Boxes:** Rectangular boxes drawn around detected objects to indicate their locations in the image or camera feed.
- **Confidence Scores:** Scores indicating the model's confidence in each detection.

If no objects belonging to the 7 supported classes are detected, no bounding boxes, labels or confidence scores are displayed. Instead, the message "No object detected" is displayed.

## 9. Troubleshooting
The following table lists common issues that may occur when running the application and their possible solutions.

| Issue    | Possible Solution |
| -------- | ------- |
| Installation of the application dependencies using `requirements.txt` fails. | Run `pip install streamlit pillow ultralytics streamlit-webrtc av` in the terminal. |
| Missing libraries when running the dataset or model scripts. | Run `pip install imagehash numpy pillow ultralytics matplotlib opencv-python` in the terminal. |
| The application does not start. |  Ensure that you are in the project's root directory and that all required dependencies are installed.   |
| streamlit is not recognised as a command. |  Install Streamlit by running `pip install streamlit` .  |
| The application does not open in the browser automatically.   | Follow the **local URL** displayed in the terminal.    |
| `ModuleNotFoundError` |  Install the missing Python module and ensure that all required dependencies listed in the [Installation](#5-installation) section are installed. |
| The camera does not work.  | Click on **SELECT DEVICE** to verify that the correct camera is selected. If the issue persists, check your browser's camera permissions and ensure that the camera is connected and available.  |
| The YOLO model cannot be found. | Ensure that the trained model file (`best.pt`) exists at the path specified by `MODEL_PATH` in `ui/config.py`.|
| Live camera detection is lagging. | Reduce `imgsz` in the `detect_live_frame` function from **640** to **320**. |

If the problem persists, check the error messages displayed in the terminal where the application was started.


### Windows: Long Path File Error
When downloading or extracting the `raw` dataset folder on Windows, you may encounter an error indicating that the file name or path is too long.

This is caused by some filenames or folder paths exceeding Window's default path length limitations.

To resolve this, follow these steps:

1. Enable long paths in Windows:
- Press `Win + R`, type `regedit` and press Enter.
- Navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`.
- Locate `LongPathsEnabled` and set its value to **1**.
- If the entry does not exist, create a new `DWORD (32-bit) Value` named `LongPathsEnabled` and set it to **1**.

2. Restart your computer for the changes to take effect.

3. Enable long paths in Git by running the following command in the terminal:
```bash
git config --global core.longpaths true
```
4. Extract the `raw` folder again. 

## 10. Accuracy & Error Analysis

### 10.1 Evaluation Metrics

### 10.2 Brief Error Analysis


## 11. Real-World Evaluation



## 12. References

**Model evaluation:**

sarahsftri. (2024) *YOLOv8 classification metrics #9946*. Available at: https://github.com/ultralytics/ultralytics/issues/9946#issuecomment-2132207985 (Accessed: 27 October 2026).

Ultralytics (n.d.) *Model Validation with Ultralytics YOLO*. Available at: https://docs.ultralytics.com/modes/val (Accessed: 27 October 2026).

**Convert YOLO bounding box coordinates:**

HansHirse. (2020) *How to convert Yolo format bounding box coordinates into OpenCV format*. Available at: https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format (Accessed: 20 October 2026).

**Model Hyperparameter Tuning:**

glenn-jocher. (2026) *tuner.py*. Available at: https://github.com/ultralytics/ultralytics/blob/v8.4.190/ultralytics/engine/tuner.py (Accessed: 26 October 2026).

**UI:**

Streamlit (n.d.) *Theming overview*. Available at: https://docs.streamlit.io/develop/concepts/configuration/theming (Accessed: 17 October 2026).