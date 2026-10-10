"""
diagnose_preprocessing.py: READ-ONLY check of why every image gives the same answer.
Does NOT change sprout_core.py. Run from the sprout folder:

    python diagnose_preprocessing.py

Tries 3 pixel scalings on your demo_images and counts how many match the filename.
"""
import glob
import os

import numpy as np
from PIL import Image

from sprout_core import Sprout

# filename keyword -> expected class index (from class_labels.json)
EXPECT = [("blight_maize", 2), ("common", 1), ("gray_leaf", 0), ("tomato_healty", 13), ("tomato_healthy", 13),
          ("healty", 3), ("healthy", 3), ("bacterial", 4), ("early_blight", 5), ("late_blight", 6),
          ("leaf mold", 7), ("leaf_mold", 7), ("septoria", 8), ("mites", 9), ("target", 10),
          ("yellow", 11), ("mosaic", 12)]
MODES = {
    "A [-1,1] (current)": lambda x: x / 127.5 - 1.0,
    "B [0,1]": lambda x: x / 255.0,
    "C [0,255] raw": lambda x: x,
}


def expected(fname):
    f = fname.lower()
    for key, idx in EXPECT:
        if key in f:
            return idx
    return None


s = Sprout()
files = sorted(p for e in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join("demo_images", f"*.{e}")))
print(f"{len(files)} images\n")
score = {m: 0 for m in MODES}
n = 0
for p in files:
    exp = expected(os.path.basename(p))
    x = np.asarray(Image.open(p).convert("RGB").resize((224, 224)), dtype=np.float32)
    row = []
    for m, f in MODES.items():
        s.interpreter.set_tensor(s.inp["index"], f(x)[None].astype(np.float32))
        s.interpreter.invoke()
        pr = s.interpreter.get_tensor(s.out["index"])[0]
        k = int(np.argmax(pr))
        row.append(f"{m[0]}:{k:2d}({pr[k]:.2f})")
        if exp is not None and k == exp:
            score[m] += 1
    if exp is not None:
        n += 1
    print(f"{os.path.basename(p):32s} expect={exp!s:>4}  " + "  ".join(row))
print("\nCorrect out of", n)
for m, v in score.items():
    print(f"  {m}: {v}")
