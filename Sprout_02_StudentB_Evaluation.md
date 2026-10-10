# SPROUT: Brief for Student B (Retrain Clean Model + Evaluation, all paper numbers)

> Give this whole file to your AI first. Also read `Sprout_00_Master_Plan_and_Lead_Tasks.md` sections 1-5. Student A builds the demo; the Lead handles the supervisor, farmers and paper writing.

## Goal
Produce every number and figure for the paper, **reproducibly**, from a **leak-free retrained model** exported as TFLite and run through `sprout_core.py`. Report honestly even if the numbers drop. Done when `results/RESULTS.md` and `results/figures/` are complete.

## What you have
- `sprout_core.py` (shared pipeline), `config.json`, `class_labels.json`, `advice_lookup.json`.
- `sprout_model.tflite`: the **original** model. Its training script was lost in a laptop crash, so its exact recipe and split are unknown. Its input is 1x224x224x3 float32, output 1x14.
- `make_split.py` and `train.py`: a rebuilt pipeline. The recipe comes from the Lead's research log (MobileNetV2, ImageNet weights, GAP, Dropout, Dense 14; phase 1 head only at lr 1e-3, phase 2 fine-tune from layer 100 at lr 1e-5; float16 TFLite export). I tested both on a small fake dataset, so the code runs end to end, but **not on the real dataset or a GPU**; expect to fix small things.
- The dataset *New Plant Diseases Dataset (Augmented)* (OneDrive; `train/` and `valid/` with class folders like `Tomato___Early_blight`). You only need the 14 tomato/maize classes in `class_labels.json`. The Lead's log reports 91.98% (phase 1) and 99.05% (phase 2) validation accuracy for the original model; treat that as a claim to re-measure, not a result.
- A ResNet notebook (`plant-disease-classification-resnet-99-2.ipynb`, probably 38 classes) for the baseline idea.

## Tasks, in order
1. **Look at the data.** Open the folders, check that the 14 class names match `class_labels.json`, and look at real file names. `make_split.py` assumes the original photo ID is the text before `___` in the file name; confirm it, or edit `group_key()`.
2. **Leakage report.** Run `make_split.py`. It writes `results/leakage_report.md` (what % of `valid/` images have a copy of the same original photo in `train/`) and `results/split_manifest.csv` (a split where all copies of one photo stay together). Check the "avg copies per original" line; if it is about 1.0, the ID rule is wrong. **Tell the Lead the leakage % the same day.**
3. **Re-measure the old model (for contrast only).** Run `sprout_model.tflite` on the original `valid/` folder through `sprout_core.py`. If it is far below about 99%, preprocessing may differ; tell the Lead. Report this as "original model, original split", never as the final result.
4. **Retrain the clean model.** Run `train.py` on a free Colab GPU (upload only the 14 class folders to Colab or Drive). Check `train_report.txt` (TFLite must agree with Keras) and `training_history.png`. Copy `sprout_model_clean.tflite` into the shared `sprout/` folder and set `"model_file": "sprout_model_clean.tflite"` in `config.json`. Tell A and the Lead. Class order is unchanged.
5. **Metrics on the clean test split.** Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix (counts and normalized, 300 dpi PNG). All via `sprout_core.py`, on the test rows of `split_manifest.csv`.
6. **One baseline.** Train a ResNet (e.g. `tf.keras.applications.ResNet50`, using its own `preprocess_input`) on the same manifest by adapting `train.py`; report accuracy, macro-F1, size, latency. Optional: EfficientNetB0.
7. **Confidence analysis.** For thresholds 0.50-0.99, plot coverage vs accuracy of accepted predictions. Report the share of predictions and the accuracy in each tier (high >= 0.85, medium >= 0.65, low). Optionally ECE and a reliability plot. Recommend final thresholds and write them to `config.json`.
8. **Robustness proxy for field conditions.** Brightness down/up, blur, JPEG compression, zoom/crop, gray or colored background padding. Report the accuracy drop and fallback rate per perturbation. Label it clearly as a *proxy*, not field data.
9. **Out-of-distribution check.** Run 30-50 non-leaf images (random noise, objects, other plants) and report what % get "high" confidence. A random-noise image already gets "Late blight, 0.95, high" from the old model, so expect the fallback to fail here. Report it as a limitation; do not hide it.
10. **Efficiency.** Model size (MB), median and p95 latency over 100 runs after warm-up on your laptop CPU (name the CPU and runtime). Note: the old `.tflite` is about 10 MB while a float16 MobileNetV2 should be about 4.5 MB, so the old file may not be float16 as the log says. Describe the new model's real size, and do not call anything float16 unless you checked.
11. **`results/RESULTS.md`:** every number, the script that made it, library versions, seeds, the date.

## Rules
- Final numbers come from the retrained `.tflite` through `sprout_core.py`, on the grouped test split only.
- Never tune thresholds or hyperparameters on the test set.
- Do not invent or round up results. A disappointing honest number beats a pretty one.
- If the Lead later provides real field photos, put them in `data/field/<class_name>/`, evaluate with the same scripts, report them **separately**, and never train on them.

## Send back
`eval/` scripts, `results/` (RESULTS.md, figures, CSVs, leakage report, split manifest), updated `config.json`, `sprout_model_clean.tflite`.
