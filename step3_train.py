import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# ── Critical for RTX 3050 4GB ──
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── fp16: cuts VRAM usage ~40% ──
mixed_precision.set_global_policy("mixed_float16")

MANIFEST        = "manifest.csv"
IMG_SIZE        = 299
BATCH_SIZE      = 4
ACCUM_STEPS     = 8
EPOCHS_HEAD     = 5
EPOCHS_FINE     = 20
LR_HEAD         = 1e-3
LR_FINE         = 1e-5
STEPS_PER_EPOCH = 1000
VAL_STEPS       = 200
AUTOTUNE        = tf.data.AUTOTUNE

def load_split(split):
    paths, labels = [], []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                paths.append(row["path"])
                labels.append(int(row["label"]))
    return paths, labels

def make_dataset(paths, labels, augment=False, batch_size=BATCH_SIZE):
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(2048).map(load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

print("Loading manifest...")
train_paths, train_labels = load_split("train")
val_paths,   val_labels   = load_split("val")
print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

train_ds = make_dataset(train_paths, train_labels, augment=True)
val_ds   = make_dataset(val_paths,   val_labels,   augment=False, batch_size=2)

# ── build model ──
base = tf.keras.applications.Xception(
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    pooling="avg"
)
base.trainable = False

x = tf.keras.layers.Dropout(0.3)(base.output)
x = tf.keras.layers.Dense(256, activation="relu", dtype="float32")(x)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
model = tf.keras.Model(base.input, out)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),
    tf.keras.callbacks.CSVLogger("training_log.csv"),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.5,
        patience=3,
        mode="max",
        verbose=1
    ),
]

# ── Phase 1: train head only ──
print("\n=== Phase 1: training head only ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_HEAD),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)

# ── Phase 2: fine-tune top blocks ──
print("\n=== Phase 2: fine-tuning top blocks ===")
base.trainable = True
for layer in base.layers:
    layer.trainable = "block14" in layer.name or "block13" in layer.name

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINE),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD + EPOCHS_FINE,
    initial_epoch=EPOCHS_HEAD,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)

model.save("deepidentity_finetuned.h5")
print("\nSaved deepidentity_finetuned.h5")
print("Check training_log.csv for per-epoch metrics")