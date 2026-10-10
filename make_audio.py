"""
make_audio.py: ONE-TIME audio generation for Sprout (needs internet, run it ONCE).

    python make_audio.py              # makes only the files that are missing
    python make_audio.py --force      # regenerates all 15

Creates audio/v1_tts/00.mp3 ... 13.mp3 (named by class index in class_labels.json)
plus low.mp3 (the retake-photo / KVK message), from advice_lookup.json, Kannada text.
The demo app (app.py) NEVER imports gTTS and never needs the internet.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "audio", "v1_tts")


def main():
    force = "--force" in sys.argv
    try:
        from gtts import gTTS
    except ImportError:
        sys.exit("Run: pip install gTTS   (needed only for this script)")

    with open(os.path.join(HERE, "class_labels.json"), encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(HERE, "advice_lookup.json"), encoding="utf-8") as f:
        advice = json.load(f)

    jobs = [(f"{i:02d}.mp3", advice[labels[i]]["kannada"], labels[i]) for i in sorted(labels)]
    jobs.append(("low.mp3", advice["_confidence_low"]["kannada"], "_confidence_low"))

    os.makedirs(OUT, exist_ok=True)
    failed = []
    for fname, text, name in jobs:
        path = os.path.join(OUT, fname)
        if os.path.exists(path) and os.path.getsize(path) > 1000 and not force:
            print(f"skip   {fname}  (exists)")
            continue
        for attempt in range(3):
            try:
                gTTS(text=text, lang="kn").save(path)
                print(f"made   {fname}  <- {name}")
                break
            except Exception as e:  # network hiccup / rate limit
                print(f"retry  {fname}: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            failed.append(fname)
        time.sleep(0.5)

    present = [f for f, _, _ in jobs if os.path.exists(os.path.join(OUT, f))]
    print(f"\n{len(present)}/15 files present in {OUT}")
    if failed:
        print("FAILED:", ", ".join(failed), "-> run the script again")
    else:
        print("Done. Now LISTEN to all 15 (names, doses, 'KVK', 'Dithane M-45' are read by a Kannada voice).")


if __name__ == "__main__":
    main()
