"""
Sprout demo app (laptop, fully offline).  Run:  streamlit run app.py

Uses sprout_core.py as-is (no preprocessing/threshold changes here).
Never imports gTTS: audio is pre-generated in audio/v1_tts/.
"""
import csv
import glob
import os
from datetime import datetime

import streamlit as st
from PIL import Image

from sprout_core import Sprout

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(HERE, "audio", "v1_tts")
SAMPLE_DIR = os.path.join(HERE, "demo_images")
LOG_PATH = os.path.join(HERE, "results", "demo_log.csv")
LOG_HEADER = ["timestamp", "image_name", "class", "confidence", "tier", "latency_ms"]

st.set_page_config(page_title="Sprout", page_icon="🌱", layout="centered")


@st.cache_resource
def load_model():
    return Sprout()


def pretty(name):
    crop, _, disease = name.partition("___")
    crop = crop.replace("_(maize)", " (maize)")
    return f"{crop}: {disease.replace('_', ' ').strip()}"


def log_run(image_name, r):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_HEADER)
        w.writerow([datetime.now().isoformat(timespec="seconds"), image_name,
                    r["class_name"], f'{r["confidence"]:.4f}', r["tier"], f'{r["latency_ms"]:.1f}'])


sprout = load_model()

st.title("🌱 Sprout")
st.caption("Offline tomato & maize leaf-disease check with Kannada voice advice. Laptop prototype.")

source = st.radio("Leaf image", ["Sample image", "Upload", "Webcam"], horizontal=True)

img, img_name = None, None
if source == "Sample image":
    files = sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(SAMPLE_DIR, f"*.{ext}")))
    if files:
        choice = st.selectbox("Choose a sample", [os.path.basename(p) for p in files])
        img_name = choice
        img = Image.open(os.path.join(SAMPLE_DIR, choice))
    else:
        st.info("No samples yet. Put 8 images in the `demo_images/` folder, or use Upload.")
elif source == "Upload":
    up = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    if up:
        img_name, img = up.name, Image.open(up)
else:
    cam = st.camera_input("Take a leaf photo")
    if cam:
        img_name, img = f"webcam_{datetime.now():%H%M%S}.jpg", Image.open(cam)

if img is not None:
    st.image(img, caption=img_name, width=320)
    if st.button("Analyse leaf", type="primary"):
        r = sprout.predict(img)
        kn, en = sprout.advice(r)
        log_run(img_name, r)
        st.session_state["res"] = dict(r=r, kn=kn, en=en, name=img_name)

res = st.session_state.get("res")
if res and res["name"] == img_name:
    r, kn, en = res["r"], res["kn"], res["en"]
    st.divider()
    tier = r["tier"]

    if tier == "low":
        st.error("⚠️ Not sure about this leaf. **Retake the photo / contact KVK.**")
    else:
        st.subheader(pretty(r["class_name"]))
        (st.success if tier == "high" else st.warning)(
            f"Confidence tier: **{tier.upper()}**" + ("" if tier == "high" else " (check with KVK if unsure)"))

    st.progress(min(max(r["confidence"], 0.0), 1.0), text=f'Confidence: {r["confidence"]*100:.1f}%')
    st.caption(f'Inference time: {r["latency_ms"]:.0f} ms  |  thresholds: high ≥ {sprout.high}, medium ≥ {sprout.medium}')

    st.markdown("**English**")
    st.write(en)
    st.markdown("**ಕನ್ನಡ (Kannada)**")
    st.write(kn)

    audio_file = os.path.join(AUDIO_DIR, "low.mp3" if tier == "low" else f'{r["class_idx"]:02d}.mp3')
    if os.path.exists(audio_file):
        st.audio(audio_file, format="audio/mp3", autoplay=True)
    else:
        st.warning(f"Audio file missing: {os.path.relpath(audio_file, HERE)}. Run make_audio.py once (needs internet).")
