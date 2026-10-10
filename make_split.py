"""
make_split.py: leakage report + leak-free (grouped) split for the 14 Sprout classes.

WHY: the 'New Plant Diseases Dataset (Augmented)' contains several edited copies of the
same original leaf photo. Its own train/ and valid/ folders were split AFTER augmenting,
so copies of one leaf can sit in both. We (1) measure that, (2) re-split so that ALL
copies of one original photo stay in the same split.

Original photo ID = the part of the file name before '___'
(e.g. '0a5e9323-dbad-...-d291718345d9___FREC_Scab 3164.JPG'). CHECK this assumption by
looking at the file names in your copy of the dataset; change group_key() if needed.

USAGE:
  python make_split.py --data_root "<folder that contains train/ and valid/>" \
                       --labels class_labels.json --out_dir results --seed 42
OUTPUT: <out_dir>/split_manifest.csv  (path,class_name,class_idx,group,split)
        <out_dir>/leakage_report.md
"""
import argparse, csv, json, os, random
from collections import defaultdict

EXTS = (".jpg", ".jpeg", ".png")


def group_key(fname):
    stem = os.path.splitext(fname)[0]
    return stem.split("___")[0] if "___" in stem else stem


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--labels", default="class_labels.json")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    with open(a.labels, encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    classes = [labels[i] for i in sorted(labels)]

    # ---- collect files from the dataset's own train/ and valid/ ----
    files = {"train": defaultdict(list), "valid": defaultdict(list)}
    for split in files:
        for c in classes:
            folder = os.path.join(a.data_root, split, c)
            imgs = list_images(folder)
            if not imgs:
                print(f"WARNING: no images in {folder}")
            files[split][c] = [(os.path.join(folder, f), group_key(f)) for f in imgs]

    # ---- leakage report on the ORIGINAL split ----
    lines = ["# Leakage report (original train/valid folders)\n",
             "| class | valid images | valid images whose original photo is also in train | % |",
             "|---|---|---|---|"]
    tot_v = tot_leak = 0
    for c in classes:
        train_keys = {k for _, k in files["train"][c]}
        v = files["valid"][c]
        leak = sum(1 for _, k in v if k in train_keys)
        tot_v += len(v); tot_leak += leak
        pct = 100.0 * leak / len(v) if v else 0.0
        lines.append(f"| {c} | {len(v)} | {leak} | {pct:.1f} |")
    overall = 100.0 * tot_leak / tot_v if tot_v else 0.0
    lines.append(f"| **ALL** | {tot_v} | {tot_leak} | **{overall:.1f}** |")
    n_imgs = sum(len(files[s][c]) for s in files for c in classes)
    n_groups = len({(c, k) for s in files for c in classes for _, k in files[s][c]})
    lines.append(f"\nTotal images: {n_imgs}; distinct original-photo IDs: {n_groups}; "
                 f"avg copies per original: {n_imgs / max(n_groups, 1):.2f}")
    lines.append("\nIf avg copies per original is ~1.0 the ID rule probably does not match "
                 "your file names: inspect them and edit group_key().")
    with open(os.path.join(a.out_dir, "leakage_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- grouped, class-stratified re-split (train+valid pooled) ----
    rng = random.Random(a.seed)
    rows = []
    for idx, c in enumerate(classes):
        by_group = defaultdict(list)
        for s in files:
            for p, k in files[s][c]:
                by_group[k].append(p)
        groups = sorted(by_group)
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * a.test_frac))
        n_val = max(1, round(n * a.val_frac))
        for i, g in enumerate(groups):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for p in by_group[g]:
                rows.append((p, c, idx, f"{c}|{g}", split))

    with open(os.path.join(a.out_dir, "split_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_name", "class_idx", "group", "split"])
        w.writerows(rows)

    counts = defaultdict(lambda: defaultdict(int))
    for _, c, _, _, s in rows:
        counts[c][s] += 1
    print("\nclean split image counts (train / val / test):")
    for c in classes:
        d = counts[c]
        flag = "   <-- few test images" if d["test"] < 30 else ""
        print(f"  {c}: {d['train']} / {d['val']} / {d['test']}{flag}")


if __name__ == "__main__":
    main()
