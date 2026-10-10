"""
sprout_core.py: the ONE shared pipeline for Sprout (laptop version).

Student A's demo app and Student B's evaluation scripts must BOTH import this
file, so the numbers in the paper come from exactly the pipeline we demo.
Do not change preprocessing here without telling everyone.

    from sprout_core import Sprout
    s = Sprout()
    r = s.predict("leaf.jpg")      # dict: class_idx, class_name, confidence, tier, probs, latency_ms
    kn, en = s.advice(r)           # Kannada text, English text

Preprocessing = MobileNetV2 style: resize 224x224, RGB, scale pixels to [-1, 1]
(same as tf.keras.applications.mobilenet_v2.preprocess_input, which the original
sprout_inference.py used). Confirm against the original train.py.
"""
import json
import os
import time

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter      # pip install ai-edge-litert
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter  # pip install tflite-runtime
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # full TensorFlow

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (224, 224)


class Sprout:
    def __init__(self, model_path=None, labels_path=None, advice_path=None, config_path=None):
        config_path = config_path or os.path.join(HERE, "config.json")
        cfg = {"high_threshold": 0.85, "medium_threshold": 0.65, "model_file": "sprout_model.tflite"}
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                cfg.update(json.load(f))
        self.high = cfg["high_threshold"]
        self.medium = cfg["medium_threshold"]

        model_path = model_path or os.path.join(HERE, cfg["model_file"])
        labels_path = labels_path or os.path.join(HERE, "class_labels.json")
        advice_path = advice_path or os.path.join(HERE, "advice_lookup.json")

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

        with open(labels_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.class_labels = {int(k): v for k, v in raw.items()}
        with open(advice_path, encoding="utf-8") as f:
            self.advice_lookup = json.load(f)

    @staticmethod
    def preprocess(img):
        """img: file path or PIL.Image -> float32 array (1, 224, 224, 3) in [-1, 1]."""
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB").resize(IMG_SIZE)
        x = np.asarray(img, dtype=np.float32)
        x = x / 127.5 - 1.0
        return np.expand_dims(x, axis=0)

    def predict_probs(self, img):
        x = self.preprocess(img)
        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out["index"])[0].copy()

    def tier_for(self, confidence):
        if confidence >= self.high:
            return "high"
        if confidence >= self.medium:
            return "medium"
        return "low"

    def predict(self, img):
        t0 = time.perf_counter()
        probs = self.predict_probs(img)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return {
            "class_idx": idx,
            "class_name": self.class_labels[idx],
            "confidence": conf,
            "tier": self.tier_for(conf),
            "probs": probs,
            "latency_ms": latency_ms,
        }

    def advice(self, result):
        """Returns (kannada_text, english_text). Low tier -> retake-photo / KVK message."""
        if result["tier"] == "low":
            entry = self.advice_lookup["_confidence_low"]
        else:
            entry = self.advice_lookup.get(result["class_name"], self.advice_lookup["_confidence_low"])
        return entry["kannada"], entry["english"]


if __name__ == "__main__":
    import sys
    s = Sprout()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    img = path or Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    r = s.predict(img)
    kn, en = s.advice(r)
    print(r["class_name"], f'{r["confidence"]:.3f}', r["tier"], f'{r["latency_ms"]:.1f} ms')
    print(en)
