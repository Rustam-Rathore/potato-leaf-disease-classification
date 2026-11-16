from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "potato_cnn_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "classification_report.txt"
IMAGE_SIZE = (128, 128)

st.set_page_config(
    page_title="PotatoAI | Leaf Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #f5f7f2;
    }
    [data-testid="stSidebar"] {
        background: #102d22;
    }
    [data-testid="stSidebar"] * {
        color: #f4f7ef !important;
    }
    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        color: #173d2c;
        letter-spacing: -0.7px;
    }
    .eyebrow {
        color: #64816e;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }
    .hero {
        background: linear-gradient(115deg, #173d2c 0%, #286044 100%);
        padding: 30px 32px;
        border-radius: 22px;
        color: white;
        margin: 10px 0 24px 0;
    }
    .hero h1 {
        font-family: 'Manrope', sans-serif;
        color: white;
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        line-height: 1.12;
        margin: 8px 0 10px 0;
    }
    .hero p {
        color: #d9e9dc;
        font-size: 1rem;
        max-width: 680px;
        margin-bottom: 0;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #e5ebe2;
        border-radius: 18px;
        padding: 22px;
    }
    .metric-label {
        color: #718276;
        font-size: .82rem;
        font-weight: 600;
    }
    .metric-value {
        color: #183d2c;
        font-family: 'Manrope', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 4px;
    }
    .result {
        background: #e8f3e9;
        border: 1px solid #cfe3d1;
        border-radius: 18px;
        padding: 22px;
    }
    .result-title {
        color: #173d2c;
        font-family: 'Manrope', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
    }
    div.stButton > button {
        background: #23613f;
        color: white;
        border: 0;
        border-radius: 11px;
        padding: .62rem 1.2rem;
        font-weight: 700;
        min-height: 46px;
    }
    div.stButton > button:hover {
        background: #174b30;
        color: white;
        border: 0;
    }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #9ab6a0;
        border-radius: 16px;
        padding: 12px;
    }
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #173d2c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        return None
    names = [
        line.strip()
        for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return names or None


def pretty_label(label: str) -> str:
    return label.replace("potato__", "").replace("_", " ").replace("-", " ").title()


def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    return image, np.expand_dims(array, axis=0)

with st.sidebar:
    st.markdown("## 🌿 PotatoAI")
    st.caption("CNN-powered leaf classification")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Detect Disease", "About Model"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("Potato Leaf Classification")
    st.caption("Built with Streamlit + TensorFlow")

model_exists = MODEL_PATH.exists()
class_names = load_class_names()

if page == "Dashboard":
    st.markdown('<div class="eyebrow">AI agriculture • overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand">PotatoAI</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow" style="color:#b8d7bf;">SMART CROP INSIGHTS</div>
            <h1>A clearer view of<br>potato leaf health.</h1>
            <p>Upload a potato leaf image and let your trained CNN model classify it.
            Get a prediction with class-wise model scores in seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="panel"><div class="metric-label">MODEL INPUT</div>'
            '<div class="metric-value">128 × 128</div><div class="metric-label">RGB image</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="panel"><div class="metric-label">OUTPUT CLASSES</div>'
            '<div class="metric-value">3</div><div class="metric-label">Softmax predictions</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        status = "Ready" if model_exists and class_names else "Files needed"
        st.markdown(
            f'<div class="panel"><div class="metric-label">MODEL STATUS</div>'
            f'<div class="metric-value">{status}</div>'
            '<div class="metric-label">Local model setup</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Get started")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(
            """
            <div class="panel">
                <h3>How it works</h3>
                <ol>
                    <li>Open <b>Detect Disease</b> from the sidebar.</li>
                    <li>Upload a clear potato leaf photo.</li>
                    <li>Run the CNN prediction.</li>
                    <li>Review the predicted class and class scores.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == "Detect Disease":
    st.markdown('<div class="eyebrow">CNN image analysis</div>', unsafe_allow_html=True)
    st.title("Detect potato leaf condition")
    st.write("Upload a leaf image in JPG, JPEG or PNG format to get a model prediction.")

    if not model_exists:
        st.error(
            "Model file not found. Place `potato_cnn_model.keras` in the same folder as `app.py`."
        )
        st.stop()
    if not class_names:
        st.error(
            "`classification_report.txt` not found or empty. Create it with one class name per line, "
            "in the exact same order used during model training."
        )
        st.stop()

    uploaded = st.file_uploader(
        "Choose a potato leaf image",
        type=["jpg", "jpeg", "png"],
        help="For best results, use a clear, well-lit image with the leaf in focus.",
    )

    if uploaded is not None:
        original, batch = prepare_image(uploaded)
        image_col, result_col = st.columns([1, 1], gap="large")

        with image_col:
            st.markdown("#### Uploaded image")
            st.image(original, use_container_width=True)
            st.caption(f"Original size: {original.width} × {original.height} px")

        with result_col:
            st.markdown("#### Analysis")
            st.markdown(
                '<div class="panel"><div class="metric-label">READY FOR CLASSIFICATION</div>'
                '<div class="metric-value">Your image is loaded</div>'
                '<p style="color:#718276;margin-bottom:0;">The image will be resized to 128 × 128 RGB pixels, '
                'matching the CNN input shape.</p></div>',
                unsafe_allow_html=True,
            )
            run_prediction = st.button("✨  Analyse leaf image", use_container_width=True)

        if run_prediction:
            try:
                model = load_model()
                scores = np.asarray(model.predict(batch, verbose=0))[0]

                if len(scores) != len(class_names):
                    st.error(
                        f"Model returned {len(scores)} scores, but the class file contains "
                        f"{len(class_names)} labels. Check the class order and files."
                    )
                    st.stop()

                best_index = int(np.argmax(scores))
                best_label = pretty_label(class_names[best_index])
                confidence = float(scores[best_index]) * 100

                st.markdown("### Prediction")
                st.markdown(
                    f"""
                    <div class="result">
                        <div class="eyebrow">TOP PREDICTED CLASS</div>
                        <div class="result-title">{best_label}</div>
                        <div style="font-size:2rem;font-weight:800;color:#23613f;margin-top:8px;">
                            {confidence:.2f}%
                        </div>
                        <div style="color:#587261;">Model confidence score</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Class-wise scores")
                for label, score in zip(class_names, scores):
                    st.write(f"**{pretty_label(label)}** — {float(score) * 100:.2f}%")
                    st.progress(float(np.clip(score, 0, 1)))

                st.info(
                    "These values are the CNN model's output scores, not a guarantee of diagnosis. "
                    "Image quality and training data can affect predictions."
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

else:
    st.markdown('<div class="eyebrow">Technical details</div>', unsafe_allow_html=True)
    st.title("About the CNN model")
    st.write("This interface uses the trained Keras model from your notebook.")

    st.markdown("### Model configuration")
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        st.markdown(
            """
            <div class="panel">
                <b>Input pipeline</b>
                <ul>
                    <li>Image size: 128 × 128</li>
                    <li>Channels: RGB</li>
                    <li>Rescaling: 1/255 inside model</li>
                    <li>Augmentation layers in model</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with config_col2:
        st.markdown(
            """
            <div class="panel">
                <b>CNN architecture</b>
                <ul>
                    <li>Conv2D: 32 filters</li>
                    <li>Conv2D: 64 filters</li>
                    <li>Conv2D: 128 filters</li>
                    <li>Conv2D: 256 filters</li>
                    <li>GlobalAveragePooling2D</li>
                    <li>Dense 128 + Dropout 0.3</li>
                    <li>Dense 3 + Softmax</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Training setup from your notebook")
    st.write(
        "Optimizer: Adam · Learning rate: 0.001 · Loss: sparse categorical crossentropy · "
        "Maximum epochs: 40 · EarlyStopping and ReduceLROnPlateau callbacks."
    )

    if class_names:
        st.markdown("### Class labels")
        for name in class_names:
            st.write(f"• {pretty_label(name)}")
    else:
        st.warning("Class labels file not found yet.")

    st.caption(
        "The notebook reports validation accuracy, not test accuracy. "
        "This app does not display an accuracy figure because the saved model and class-label file "
        "alone do not provide a verified evaluation metric."
    )
