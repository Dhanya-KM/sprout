# SPROUT: Master Plan (read this first, everyone)

This replaces all earlier briefs. Old versions (web app/PWA, 38 classes, Grad-CAM, severity) are cancelled.

## 1. What Sprout is
An offline tool that tells a farmer which disease a tomato or maize leaf has, how sure it is, and what to do, spoken in **Kannada**. The model is MobileNetV2 (transfer learning, 14 classes: 4 maize + 10 tomato incl. healthy), exported as TFLite. Pipeline: photo -> model -> confidence tier -> pre-written advice (brand, grams per 10 L water, KVK fallback) -> Kannada audio. Low confidence -> "retake the photo or contact KVK" instead of a guess.

## 2. Decisions (locked, do not reopen)
- **Software only, runs on a laptop.** No Raspberry Pi, no web app, no phone app.
- **14 classes** (tomato + maize). Nothing else.
- **The original `train.py`, `.h5` and dataset script are lost (laptop crash). We retrain one clean model** with `make_split.py` + `train.py` (included). Until it is ready, the demo uses the original `sprout_model.tflite`.
- **Cut:** Grad-CAM, severity scoring, 4-model comparison, dialect TTS research, genomics, weather, LLM text.
- The paper is about **evidence**, not about the app being new (offline voice disease apps already exist).

## 3. The shared goal (all three of us)
By the end we have: **(1)** a demo that runs offline on a laptop, **(2)** an honest, leakage-checked evaluation package, **(3)** a paper draft, and **(4, stretch)** a small farmer comprehension pilot if the supervisor approves.
Test for any task: *does it end up in the demo for ma'am or in the paper?* If not, don't do it.

## 4. Who does what
| Person | Goal | Brief file |
|---|---|---|
| **Student A** | Working offline demo + audio + logging (+ study mode later) | `Sprout_01_StudentA_Demo.md` |
| **Student B** | Every number and figure for the paper, from the deployed model | `Sprout_02_StudentB_Evaluation.md` |
| **Lead (you)** | Supervisor sign-off, advice and audio content, farmer pilot, paper writing | Section 7 below |

## 5. Shared contract (so nobody's work conflicts)
- One folder, `sprout/`: `sprout_core.py`, `config.json`, `sprout_model.tflite`, `class_labels.json`, `advice_lookup.json`, `audio/` (A), `app.py` (A), `eval/` (B), `results/` (B, plus A's `demo_log.csv`).
- **`sprout_core.py` is the single pipeline.** A's demo and B's evaluation both import it. Final paper numbers must come from the deployed `.tflite` through this file. It already runs (smoke-tested: loads the model, 14 outputs, no NaN).
- **Thresholds** live only in `config.json` (now 0.85 high / 0.65 medium, from the original script). B recommends final values; A's app reads them from there.
- **Class order** = `class_labels.json` (0-13). Audio files are named by that index (`00.mp3` ... `13.mp3`, `low.mp3`).
- **Two model files:** `sprout_model.tflite` = original (recipe lost, split probably leaky; used for the demo until the new one exists). `sprout_model_clean.tflite` = retrained on the leak-free split = **the paper's model**. Switch by changing `model_file` in `config.json`; class order is identical, so audio and advice stay valid.
- Blocked more than half a day? Tell the Lead the same day.

## 6. Milestones
| # | What | Owner | Rough effort | Done when |
|---|---|---|---|---|
| M1 | Demo for ma'am | A (+ Lead prepares talk) | 1-2 days | Offline demo runs on 8 test images; 1-2 min video recorded |
| M2 | Evidence package, incl. retraining the clean model on a Colab GPU | B | 3-5 days | `results/RESULTS.md` + figures complete |
| M3 | Farmer pilot (stretch) | Lead + A | 1-2 days prep, then sessions | Only after ma'am approves and ethics/consent are cleared |
| M4 | Paper draft (4-6 pages) | Lead | 2-3 days of writing, in parallel from day 1 | Draft uses M2 numbers and M1 screenshots |

These are planning guesses, not promises. The biggest unknown is whether B's leakage check forces a retrain.

## 7. Lead tasks
1. **Today:** send the files, `make_split.py`, `train.py` and the dataset (OneDrive link; B only needs the 14 tomato/maize classes) to A and B. Nothing else is needed from your crashed laptop.
2. **Meet ma'am.** Say this:
   - *Problem:* smallholder farmers in Karnataka rarely get timely, correct disease advice. Expert visits are rare, and apps assume internet, smartphone literacy and standard-language comprehension.
   - *Built:* an offline 14-class tomato + maize leaf-disease detector with dosage-specific Kannada voice advice and a low-confidence fallback to the KVK.
   - *Model:* MobileNetV2 transfer learning in two phases (head only, about 92% validation; fine-tuned from layer 100, 99.05%), TFLite float16.
   - *Honest caveats:* the 99% is on PlantVillage-style lab images and may include duplicate augmented images across train and validation, which we are checking. Hardware (Pi) is out of scope; this is a laptop prototype.
   - *Asks:* approval to run a small farmer comprehension pilot, who handles ethics/consent, and which venue and deadline to target.
3. **Audio mapping:** listen to your 15 existing files (`0001`-`0015`) and write down which advice each one is. Tell A (expected: 14 classes + one low-confidence message, but unverified).
4. **Advice review:** have the 14 advice texts (doses, products, intervals) checked by an agronomist, KVK or your parents' local shop. One thing to check: the mosaic virus Kannada text says "jigi huLa" while the English says aphids; confirm the right word.
5. **Pilot prep (stretch):** record the advice clips in your own Tumkur speech (if you want the dialect comparison); draft the Kannada consent script and 4-5 comprehension questions (e.g. "what will you spray, how much, how often?").
6. **Paper:** write Introduction, Related Work and Limitations first. Cite only papers you have opened (your log has Nyakuri 2025 and Madiwal 2025). Do not write "first system to..." claims; offline voice-guided disease apps already exist (for example AgroAssist+ 2025 for Kannada voice + CNN), so position Sprout as an evaluation-focused prototype.
7. **Venue:** get a venue and deadline from ma'am; avoid generic conference-alert sites.

## 8. Honest limits (put these in the paper)
The original model's training recipe is lost, so the paper reports the retrained model; the old 99.05% can only be re-measured by running the old `.tflite`, and its training/validation split is probably leaky; PlantVillage images are lab-style; laptop results are not device results; the pilot (if done) is small; the model is overconfident on non-leaf inputs (see B's brief). No publication is guaranteed.
