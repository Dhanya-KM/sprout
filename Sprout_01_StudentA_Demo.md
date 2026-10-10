# SPROUT: Brief for Student A (Demo App + Audio)

> Give this whole file to your AI first. Also read `Sprout_00_Master_Plan_and_Lead_Tasks.md` sections 1-5. Student B does the evaluation; the Lead handles supervisor/farmers/paper.

## Goal
A **laptop app that runs fully offline** and shows the whole Sprout flow for the supervisor, plus a log file the paper can use. Done when the Lead can unplug Wi-Fi and demo it in 2 minutes.

## What you have
`sprout_core.py` (shared pipeline, already runs), `config.json` (thresholds and which model file to use), `sprout_model.tflite` (the original model; B will later give you `sprout_model_clean.tflite`, and you only change `model_file` in `config.json`) (input 1x224x224x3 float32, output 1x14 probabilities; checked), `class_labels.json`, `advice_lookup.json` (Kannada + English text for 14 classes + `_confidence_low`), and the Lead's 15 audio files (`0001`-`0015`, mapping to be confirmed by the Lead).
The old `sprout_inference.py` is a Raspberry Pi script. It needs the internet (gTTS at runtime) and camera/GPIO, so **do not reuse it**; use `sprout_core.py`.

## Tasks, in order
1. **Setup.** Python 3.10-3.12, `pip install ai-edge-litert pillow numpy streamlit` (or tflite-runtime / TensorFlow). Run `python sprout_core.py some_leaf.jpg` and confirm a sensible class and no NaN.
2. **Audio.** Create `audio/v1_tts/00.mp3 ... 13.mp3, low.mp3` (15 files named by class index). Use the Lead's existing files if the Lead confirms the mapping; otherwise generate once from `advice_lookup.json` with gTTS (`lang="kn"`). This needs the internet **once, at generation time only**. The app must never call gTTS while running.
3. **App** (`app.py`, Streamlit or Gradio): choose a sample image / upload / webcam; show the image, predicted disease, confidence bar, tier (high/medium/low), English advice, Kannada text (check that Kannada glyphs render properly), autoplay audio, and the inference time. Low tier shows the retake / KVK message and plays `low.mp3`.
4. **Logging.** Append every run to `results/demo_log.csv` (timestamp, image name, class, confidence, tier, latency_ms).
5. **Offline test.** Turn Wi-Fi off, restart the app, run the full flow. Fix anything that needs the network.
6. **Demo images.** 8 images: 2 healthy, 4 diseased (both crops), 1 blurry, 1 non-leaf. Take them from B's clean test list (`results/test_split.csv`) once it exists; until then use any validation images. **Known issue:** a random-noise image already gets "Late blight, 95%, high" from the model, so the fallback does NOT reliably catch non-leaf inputs. Do not claim it does; show the real behaviour and tell B.
7. **Hand-off.** README with a one-command run; 1-2 minute screen recording (select leaf, result, Kannada audio, then a low-confidence case); list of screenshots for the paper.

## Stretch (only after the Lead says "approved"): study mode
Voice versions as folders `audio/v1_tts`, `v2_dialect_tts`, `v3_human` (v2/v3 come from the Lead). A page where the facilitator enters a participant ID; the app plays the three versions in a randomized order stored in the log, and the facilitator ticks comprehension answers and trust rating per version. Save to `results/study_log.csv`. No personal data beyond an ID.

## Rules
- Do not change preprocessing or thresholds in `sprout_core.py`; ask B/Lead.
- Never claim "runs on a Pi or phone"; it is a laptop prototype.
- Do not add features beyond the list above.

## Send back
`app.py`, `audio/`, README, demo video, `demo_log.csv`, screenshots.
