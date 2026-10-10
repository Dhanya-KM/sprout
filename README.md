# Sprout demo (laptop prototype, fully offline)

## Folder layout (put these files INTO the shared `sprout/` folder)
sprout/
  sprout_core.py  config.json  class_labels.json  advice_lookup.json  sprout_model.tflite   (from the Lead)
  app.py  make_audio.py  .streamlit/config.toml                                              (from this package)
  demo_images/   <- 8 images (2 healthy, 4 diseased both crops, 1 blurry, 1 non-leaf)
  audio/v1_tts/  <- 00.mp3 ... 13.mp3, low.mp3  (made by make_audio.py)
  results/demo_log.csv  <- created automatically

## One-time setup (internet needed)
pip install ai-edge-litert pillow numpy streamlit gTTS
python sprout_core.py demo_images/<any_leaf>.jpg      # sensible class, no errors/NaN
python make_audio.py                                  # makes the 15 mp3 files; listen to all of them

## Run (no internet needed)
streamlit run app.py

## Notes
- app.py never imports gTTS. gTTS is used only by make_audio.py.
- Thresholds live in config.json (not in the app). To switch models later, change only `model_file`.
- Known issue: the model gives confident wrong answers on non-leaf images (random noise -> "Late blight, ~94%, high"),
  so the low-confidence fallback does NOT reliably catch non-leaf inputs. The demo shows this honestly. Tell Student B.
- This is a laptop prototype. It does not run on a Pi or phone.
