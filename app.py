import io
import html
import time
import base64

import streamlit as st
from PIL import Image, ImageOps

from ultralytics import YOLO

import av
import threading
from streamlit_webrtc import webrtc_streamer

from ui.styles import CSS
from ui.config import MODEL_PATH, APP_TITLE, CLASSES, show_model_info

# configure the page
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.html(f"<style>{CSS}</style>")

STAGES = ["Preparing image", "Running the model", "Drawing detections"]
state = st.session_state
state.setdefault("mode", "Upload Image")
state.setdefault("upload_key", 0) 


@st.cache_resource   
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))

# to prevent concurrent sessions from interfering with predections on same cached model
@st.cache_resource
def get_model_lock():
    return threading.Lock()

def analyse(image):
    model = load_model()

    with get_model_lock():
        result = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            verbose=False
        )[0]

    detections = []
    boxes = result.boxes

    # skip images with no detections
    if boxes is None or len(boxes) == 0:
        return [], image.copy()

    for box in boxes:
        class_id = int(box.cls[0].item())
        category = result.names[class_id]
        confidence = float(box.conf[0].item()) * 100

        detections.append({
            "category": category,
            "confidenceScore": confidence
        })

    # sort in order of confidence score
    sorted_detections = sorted(detections, key=lambda d: d['confidenceScore'], reverse=True)

    annotated_bgr = result.plot()
    annotated = Image.fromarray(
        annotated_bgr[:, :, ::-1].copy()
    )

    return sorted_detections, annotated

def detect_live_frame(frame, model, model_lock):
    # convert video frame to array
    image_bgr = frame.to_ndarray(format="bgr24")

    with model_lock:
        result = model.predict(
            source=image_bgr,
            imgsz=320,
            conf=0.25,
            verbose=False
        )[0]

    # draw bounding boxes
    annotated_bgr = result.plot()

    # return processed frame to live video
    return av.VideoFrame.from_ndarray(
        annotated_bgr,
        format="bgr24"
    )

# converts image to html
def image_to_html(img, css_class=""):
    image_buffer = io.BytesIO()
    img.save(image_buffer, format="JPEG", quality=85)
    image_data = base64.b64encode(image_buffer.getvalue()).decode()
    return f'<img class="{css_class}" src="data:image/jpeg;base64,{image_data}"/>'

# scan animation
def scan_html(img, current_stage):
    stages = "".join(
        f'<span class="process-stage {"now" if i == current_stage else ""}">{s}</span>' for i, s in enumerate(STAGES)
    )
    return (
        f'<div class="image-scan-wrap">'
            f'<div class="image-scan">'
                f'{image_to_html(img)}'
                f'<div class="image-scan-line"></div>'
            f'</div>'
            f'<div class="process-stages">{stages}</div>'
        f'</div>'
        )

# header with the title and the info button
_, title_col, info_col = st.columns([1, 8, 1], vertical_alignment="center")
title_col.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
if info_col.button("Info", key="info_btn", icon=":material/info:", help="About the model"):
    show_model_info()

# panel with the classes
class_items = "".join(
    f'<div class="class-item">&middot {html.escape(item)}</div>' 
    for item in CLASSES
)
st.markdown(
    f'<div class="classes-panel">'
    f'<h4>Detect the following Office Items with our Trained Model</h4>'
    f'<div class="classes-grid">{class_items}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# choose upload or camera
st.markdown(
    '<div class="upload-title">Provide Image for Object Detection</div>', 
    unsafe_allow_html=True
)
upload_col, camera_col = st.columns(2, gap="large")
if upload_col.button("Upload Image", key="mode_upload", type="primary" if state.mode == "Upload Image" else "secondary"):
    state.mode = "Upload Image"
    st.rerun()
if camera_col.button("Live Camera", key="mode_camera", type="primary" if state.mode == "Live Camera" else "secondary"):
    state.mode = "Live Camera"
    st.rerun()
st.write("")

# image input
file = None
camera_mode = None

if state.mode == "Upload Image":
    with st.container(key="uploader"):
        file = st.file_uploader(
            "Upload an image", 
            type=["jpg", "jpeg", "png"],
            key=f"upload_{state.upload_key}", 
            label_visibility="collapsed"
        )
else:
    camera_mode = st.radio(
        "Camera option",
        ["Live Detection", "Take a Photo"],
        horizontal=True,
        label_visibility="collapsed",
        key="camera_input_mode"
    )

    if camera_mode == "Take a Photo":
        file = st.camera_input(
            "Take a photo",
            key=f"camera_{state.upload_key}",
            label_visibility="collapsed"
        )
        if file is None:
            state.pop("result", None) # clear previous results

    else:
        state.pop("result", None)

        live_model = load_model()
        model_lock = get_model_lock()

        # continuous live object detection
        webrtc_streamer(
            key="live-object-detection",
            video_frame_callback=lambda frame: detect_live_frame(
                frame,
                live_model,
                model_lock
            ),
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            rtc_configuration={
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }
        )

photo_mode = (
    state.mode == "Upload Image"
    or camera_mode == "Take a Photo"
)   

image = None
input_error = None

if file is not None:
    try:
        image = Image.open(file)
        image = ImageOps.exif_transpose(image).convert("RGB") # correct orientation 
        image.thumbnail((1600, 1600))
    except Exception:
        input_error = "This file could not be read as an image. Use a JPG, JPEG or PNG file."

# analyse and clear buttons
st.write("")
analyse_clicked = False
clear_clicked = False
if photo_mode:
    st.write("")

    _, analyse_col, clear_col, _ = st.columns(
        [0.3, 1, 1, 0.3]
    )

    analyse_clicked = analyse_col.button(
        "Analyse Image",
        key="analyse",
        type="primary",
        disabled=image is None
    )

    clear_clicked = clear_col.button(
        "Clear Image",
        key="clear",
        disabled=image is None
    )

if clear_clicked:
    state.upload_key += 1
    state.pop("result", None)
    st.rerun()

# image analysis section
title = '<div class="upload-title" id="analysis-section">Image Analysis</div>'

if analyse_clicked:
    st.markdown(title, unsafe_allow_html=True)
    
    with st.container(key="analysis_box"):
        result_area = st.empty()

        if image is None:
            error = input_error or "No image provided. Upload an image before analysing."   
            result_area.error(error, icon=":material/error:")
            state.result = {"error": error}
            
        else:
            # animation while the model runs
            result_area.markdown(
                scan_html(image, 0), 
                unsafe_allow_html=True
            )

            time.sleep(0.4)

            result_area.markdown(
                scan_html(image, 1), 
                unsafe_allow_html=True
            )

            try:
                detections, annotated_image = analyse(image)

                state.result = {
                    "detections": detections, 
                    "image": annotated_image
                }

                result_area.markdown(
                    image_to_html(annotated_image, "analysis-image image-reveal"), 
                    unsafe_allow_html=True
                )

            except Exception as e:
                state.result = {"error": str(e)}
                result_area.error(str(e), icon=":material/error:")

    # scroll down to the results
    st.iframe(
        """
        <script>
        window.parent.document.getElementById('analysis-section')
            .scrollIntoView({
                behavior:'smooth',
                block:'start'
                });
        </script>
        """,
        height=1
    )

elif photo_mode and "result" in state:
    st.markdown(title, unsafe_allow_html=True)
    
    with st.container(key="analysis_box"):
        if "error" in state.result:
            st.error(state.result["error"], icon=":material/error:")
        else:
            st.markdown(
                image_to_html(state.result["image"], "analysis-image"), 
                unsafe_allow_html=True
            )

# classification result
result = state.get("result")
if photo_mode and result and "error" not in result:
    detections = result["detections"]
    if detections:
        results_html= ""

        # display all detections
        for index, item in enumerate(detections, start=1):
            confidence_score = item["confidenceScore"]
            if confidence_score >= 85:
                confidence_label = "High confidence"
                confidence_class="high"
            elif confidence_score >= 60:
                confidence_label = "Moderate confidence"
                confidence_class="moderate"
            else:
                confidence_label = "Low confidence"
                confidence_class="low"

            results_html += (
                f'<div class="class-result-item">'
                f'<div class="class-result-label">{index}. {html.escape(item["category"])}</div>'
                f'<div class="confidence-row"><span>Confidence Level</span><span>{confidence_score:.1f}%</span></div>'
                f'<div class="confidence-bar"><div class="confidence-bar-fill" style="width:{confidence_score:.1f}%"></div></div>'
                f'<div style="text-align:center"><span class="confidence-level {confidence_class}">{confidence_label}</span></div>'
                f'</div>'
            )
    else:
        results_html= (f'<div class="class-result-label" style="text-align:center">No object detected</div>')

    st.markdown(
        f'<div class="classification-result">'
        f'<div class="classification-result-title">Detection Results</div>'
        f'{results_html}'
        f'</div>',
        unsafe_allow_html=True,
    )