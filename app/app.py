import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bananasense.config import (  # noqa: E402
    CLASS_NAMES,
    CONFUSION_MATRIX_PATH,
    HISTORY_PATH,
    LABELS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    TEST_DIR,
    TRAIN_DIR,
)
from src.bananasense.data import count_images  # noqa: E402
from src.bananasense.predict import predict_image  # noqa: E402


st.set_page_config(page_title="BananaSense Dashboard", page_icon="🍌", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #fffaf0; }
    .hero { background: linear-gradient(135deg, #fff0a8, #ffc928); border-radius: 28px; padding: 1.8rem 2.2rem; color: #4c3500; }
    .hero h1 { color: #4c3500; font-size: 2.7rem; margin: 0; }
    .hero p { font-size: 1.05rem; margin: .4rem 0 0; }
    .banana { font-size: 4.3rem; line-height: 1; filter: drop-shadow(0 7px 4px #b8862455); }
    .panel { background: #ffffff; border: 1px solid #f0dda0; border-radius: 20px; padding: 1.2rem; box-shadow: 0 8px 24px #d5b44a22; }
    .pill { display: inline-block; background: #fff2a8; border-radius: 999px; padding: .35rem .75rem; margin: .15rem; color: #664d00; font-weight: 600; }
    .formula { background: #fff8d8; color: #604900; border-radius: 14px; padding: 1rem; font-size: 1.3rem; text-align: center; font-family: monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)


st.markdown(
    '<div class="hero"><div class="banana">🍌</div><h1>BananaSense Dashboard</h1><p>Klasifikasi kematangan pisang menggunakan Convolutional Neural Network</p></div>',
    unsafe_allow_html=True,
)
st.write("")

labels = load_json(LABELS_PATH, CLASS_NAMES)
model = load_model()
train_counts = count_images(TRAIN_DIR)
test_counts = count_images(TEST_DIR)
metrics = load_json(METRICS_PATH, {})
history = load_json(HISTORY_PATH, {})

metric_cols = st.columns(4)
metric_cols[0].metric("📚 Data train", sum(train_counts.values()))
metric_cols[1].metric("🧪 Data test", sum(test_counts.values()))
metric_cols[2].metric("🎯 Akurasi test", f"{metrics.get('test_accuracy', 0):.1%}" if metrics else "Belum dilatih")
model_status = "Belum dilatih"
if model:
    model_status = "Oke" if metrics.get("test_accuracy", 0) >= 0.85 else "Perlu tuning"
metric_cols[3].metric("🧠 Status model", model_status)

prediction_tab, training_tab, gradient_tab = st.tabs(["🍌 Prediksi", "📈 Performa Model", "🧭 Gradient Descent"])

with prediction_tab:
    upload_col, result_col = st.columns([1, 1], gap="large")
    with upload_col:
        st.markdown('<div class="panel"><h3>📸 Input citra pisang</h3></div>', unsafe_allow_html=True)
        input_mode = st.radio("Pilih sumber gambar", ["Upload file", "Kamera"], horizontal=True)
        uploaded_file = None
        if input_mode == "Upload file":
            uploaded_file = st.file_uploader(
                "Pilih file gambar", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
            )
        else:
            uploaded_file = st.camera_input("Ambil foto pisang")
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Citra masukan", use_container_width=True)
    with result_col:
        st.markdown('<div class="panel"><h3>✨ Hasil analisis</h3></div>', unsafe_allow_html=True)
        if uploaded_file and model:
            if st.button("Analisis Sekarang", type="primary", use_container_width=True):
                best, results = predict_image(uploaded_file.getvalue(), model, labels)
                st.success(f"Prediksi: {best['label'].replace('-', ' ').title()}")
                st.metric("Confidence", f"{best['probability']:.2%}")
                st.write("Probabilitas setiap kelas")
                st.bar_chart({item["label"]: item["probability"] for item in results})
        elif not model:
            st.info("Model belum tersedia. Jalankan training terlebih dahulu.")
        else:
            st.info("Upload gambar untuk melihat prediksi.")
    if metrics:
        st.markdown("### Ringkasan kualitas model")
        quality_cols = st.columns(4)
        quality_cols[0].metric("Epoch terbaik", metrics.get("best_epoch", "-"))
        quality_cols[1].metric("Val accuracy terbaik", f"{metrics.get('best_val_accuracy', 0):.1%}")
        quality_cols[2].metric("Test accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
        quality_cols[3].metric("Epoch selesai", metrics.get("epochs_completed", "-"))
        if metrics.get("test_accuracy", 0) >= 0.85:
            st.success("Model memenuhi indikator awal dengan akurasi test minimal 85%.")
        else:
            st.warning("Model belum mencapai akurasi test 85%. Tambah data, tuning, atau lanjutkan eksperimen.")
    st.markdown("### Kelas yang dikenali")
    st.markdown("".join(f'<span class="pill">{label.replace("-", " ").title()}</span>' for label in labels), unsafe_allow_html=True)

with training_tab:
    if not history:
        st.info("Grafik akan muncul setelah training selesai.")
    else:
        epochs = list(range(1, len(history.get("loss", [])) + 1))
        chart_data = {
            "epoch": epochs,
            "loss": history.get("loss", []),
            "val_loss": history.get("val_loss", []),
            "accuracy": history.get("accuracy", []),
            "val_accuracy": history.get("val_accuracy", []),
        }
        st.subheader("Loss dan accuracy per epoch")
        st.line_chart(chart_data, x="epoch", y=["loss", "val_loss"])
        st.line_chart(chart_data, x="epoch", y=["accuracy", "val_accuracy"])

        figure, axis = plt.subplots(figsize=(8, 3.5))
        axis.bar(labels, [train_counts.get(label, 0) for label in labels], color="#f4bd27", label="Train")
        axis.bar(labels, [test_counts.get(label, 0) for label in labels], color="#78b7c9", alpha=.85, label="Test")
        axis.set_ylabel("Jumlah citra")
        axis.set_title("Distribusi dataset")
        axis.legend()
        axis.tick_params(axis="x", rotation=20)
        st.pyplot(figure, use_container_width=True)

with gradient_tab:
    st.markdown('<div class="formula">θₜ₊₁ = θₜ − η × ∇J(θₜ)</div>', unsafe_allow_html=True)
    st.write("Setiap epoch, CNN menghitung gradien loss terhadap bobot, lalu Adam memperbarui bobot ke arah loss yang lebih kecil.")
    gradient_norms = history.get("gradient_norm", [])
    if gradient_norms:
        epochs = list(range(1, len(gradient_norms) + 1))
        gradient_data = {"epoch": epochs, "gradient_norm": gradient_norms}
        st.subheader("Perubahan besar gradien")
        st.line_chart(gradient_data, x="epoch", y="gradient_norm")
        st.caption("Gradient norm yang cenderung turun menunjukkan pembelajaran mulai stabil; naik-turun kecil masih normal.")
    else:
        st.info("Visualisasi gradient descent akan muncul setelah training selesai.")

    if CONFUSION_MATRIX_PATH.exists():
        matrix = load_json(CONFUSION_MATRIX_PATH, [])
        figure, axis = plt.subplots(figsize=(7, 5))
        sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap="YlOrBr", xticklabels=labels, yticklabels=labels, ax=axis)
        axis.set_xlabel("Prediksi")
        axis.set_ylabel("Label sebenarnya")
        axis.set_title("Confusion matrix data test")
        st.pyplot(figure, use_container_width=True)
