"""
train.py: retrain the Sprout 14-class MobileNetV2 on the leak-free split (new, reproducible).

Recipe (rebuilt from the Lead's research log, the original script was lost):
  MobileNetV2 (ImageNet weights) -> GlobalAveragePooling -> Dropout -> Dense(14, softmax)
  Phase 1: base frozen, head only, Adam 1e-3
  Phase 2: unfreeze base from layer 100, Adam 1e-5, keep best checkpoint
  Preprocessing: tf.keras.applications.mobilenet_v2.preprocess_input (pixels -> [-1, 1])
  Export: TFLite float16  (same input/output format as the old sprout_model.tflite)

USAGE (Colab GPU is fine):
  python make_split.py --data_root <dataset> --labels class_labels.json --out_dir results
  python train.py --manifest results/split_manifest.csv --labels class_labels.json --out_dir results
OUTPUT in out_dir: sprout_model_clean.keras, sprout_model_clean.tflite, training_history.csv/.png,
                   train_report.txt (test accuracy + TFLite-vs-Keras agreement)
"""
import argparse, csv, json, os
import numpy as np
import tensorflow as tf

IMG = 224
AUTOTUNE = tf.data.AUTOTUNE


def read_manifest(path):
    rows = {"train": [], "val": [], "test": []}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["split"]].append((r["path"], int(r["class_idx"])))
    return rows


def make_ds(items, batch, train, augment, seed):
    paths = tf.constant([p for p, _ in items])
    labels = tf.constant([l for _, l in items], dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(items), seed=seed, reshuffle_each_iteration=True)

    def load(p, l):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG))
        img.set_shape((IMG, IMG, 3))
        return tf.cast(img, tf.float32), l

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch)
    if train and augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", default="class_labels.json")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--epochs1", type=int, default=15)
    ap.add_argument("--epochs2", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true", help="smoke tests only")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(a.seed)

    with open(a.labels, encoding="utf-8") as f:
        n_classes = len(json.load(f))
    rows = read_manifest(a.manifest)
    print({k: len(v) for k, v in rows.items()})

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
    ])
    train_ds = make_ds(rows["train"], a.batch, True, augment, a.seed)
    val_ds = make_ds(rows["val"], a.batch, False, None, a.seed)
    test_ds = make_ds(rows["test"], a.batch, False, None, a.seed)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False,
        weights=None if a.no_pretrained else "imagenet")
    base.trainable = False
    inp = tf.keras.Input((IMG, IMG, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)

    ckpt = os.path.join(a.out_dir, "best.weights.h5")
    cbs = lambda: [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True,
                                           save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    ]
    hist = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    def run(epochs, lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs())
        for k in hist:
            hist[k] += h.history[k]

    print("== Phase 1: head only ==")
    run(a.epochs1, 1e-3)
    print("== Phase 2: fine-tune from layer 100 ==")
    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False
    run(a.epochs2, 1e-5)
    if os.path.exists(ckpt):
        model.load_weights(ckpt)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"TEST accuracy (leak-free split): {test_acc:.4f}")

    model.save(os.path.join(a.out_dir, "sprout_model_clean.keras"))
    with open(os.path.join(a.out_dir, "training_history.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch"] + list(hist))
        for i in range(len(hist["loss"])):
            w.writerow([i + 1] + [hist[k][i] for k in hist])
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(hist["accuracy"], label="train"); ax[0].plot(hist["val_accuracy"], label="val")
        ax[0].set_title("accuracy"); ax[0].legend()
        ax[1].plot(hist["loss"], label="train"); ax[1].plot(hist["val_loss"], label="val")
        ax[1].set_title("loss"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "training_history.png"), dpi=200)
    except Exception as e:
        print("plot skipped:", e)

    # ---- TFLite float16 export + check it agrees with the Keras model ----
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl_path = os.path.join(a.out_dir, "sprout_model_clean.tflite")
    with open(tfl_path, "wb") as f:
        f.write(conv.convert())
    it = tf.lite.Interpreter(model_path=tfl_path); it.allocate_tensors()
    i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
    xb, _ = next(iter(test_ds))
    agree, maxdiff = 0, 0.0
    kp = model.predict(xb, verbose=0)
    for j in range(len(xb)):
        it.set_tensor(i_d["index"], xb[j:j + 1].numpy()); it.invoke()
        tp = it.get_tensor(o_d["index"])[0]
        agree += int(np.argmax(tp) == np.argmax(kp[j]))
        maxdiff = max(maxdiff, float(np.abs(tp - kp[j]).max()))
    rep = (f"test_accuracy_keras={test_acc:.4f}\n"
           f"tflite_vs_keras_agreement={agree}/{len(xb)} (max prob diff {maxdiff:.4f})\n"
           f"tflite_size_MB={os.path.getsize(tfl_path) / 1e6:.2f}\n")
    open(os.path.join(a.out_dir, "train_report.txt"), "w").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
