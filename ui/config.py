import html
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "weights" / "best.pt"
GRAPHS = ROOT / "runs" / "detect" / "train"
APP_TITLE = "Office Items Classifier"
CLASSES = ["Laptop", "Keyboard", "Mouse", "Pendrive", "Mug", "Bottle", "Phone", "Bin"]

# model popup information
MODEL_INFO = {
    "task": "This image-classification model predicts the category of an office item from an uploaded image or a live camera capture.",
    "uses": "a YOLOv8 model (Ultralytics) fine-tuned on a merged office-item dataset, "
            "split 70/20/10 into train/validation/test. Each prediction comes with a "
            "bounding box and a confidence score.",
    # scores
    "metrics": {
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1 Score": None,
        "mAP@50": None,
    },
    "graphs": [
        ("Confusion matrix", GRAPHS / "confusion_matrix.png"),
        ("Training curves", GRAPHS / "results.png"),
    ],
}

def format_class_names():
    return ", ".join(CLASSES[:-1]) + f" and {CLASSES[-1]}"

@st.dialog("About the Model", width="large")
def show_model_info():
    st.markdown(
        '<div class="sub-heading">Model Overview</div>', 
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="model-info-body">{html.escape(MODEL_INFO["task"])}<br>'
        f'Classes ({len(CLASSES)}): {html.escape(format_class_names())}.<br>'
        f'It uses {html.escape(MODEL_INFO["uses"])}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-heading">Model Performance</div>', 
        unsafe_allow_html=True
    )

    metrics = MODEL_INFO["metrics"]

    metric_cards = "".join(
        f'<div class="metric-card">'
        f'<b>{"—" if score is None else f"{score * 100:.1f}%"}</b>'
        f'<span>{html.escape(metric_name)}</span>'
        f'</div>'
        for metric_name, score in metrics.items()
    )

    st.markdown(
        f'<div class="metrics-grid" style="margin-top:.4rem">{metric_cards}</div>', 
        unsafe_allow_html=True
    )

    if any(score is None for score in metrics.values()):
        st.caption("Scores are empty")

    graph_columns = st.columns(len(MODEL_INFO["graphs"]))

    for col, (graph_title, graph_path) in zip(graph_columns, MODEL_INFO["graphs"]):
        with col:
            if graph_path.exists():
                st.image(str(graph_path), caption=graph_title)
            else:
                missing_file = html.escape(
                    str(graph_path.relative_to(ROOT))
                )
                st.markdown(
                    f'<div class="chart-placeholder">'
                    f'{html.escape(graph_title)}<br>'
                    f'Missing file: {missing_file}'
                    f'</div>',
                    unsafe_allow_html=True,
                )