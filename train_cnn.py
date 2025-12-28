import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------- CONFIG ----------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 32        # ⚡ Best for CPU (NOT 64)
EPOCHS = 12
TRAIN_DIR = r"D:\dataset\train"
TEST_DIR  = r"D:\dataset\test"
MODEL_PATH = "deepfake_mobilenetv2_final.h5"

# ---------------- DATASET ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class names:", train_ds.class_names)
# MUST be: ['Fake', 'Real']

AUTOTUNE = tf.data.AUTOTUNE

# ---------------- LIGHT AUGMENTATION ----------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.03),
])

def preprocess(x, y, training=False):
    x = tf.cast(x, tf.float32) / 255.0
    if training:
        x = data_augmentation(x)
    return x, y

train_ds = train_ds.map(
    lambda x, y: preprocess(x, y, True),
    num_parallel_calls=AUTOTUNE
)

test_ds = test_ds.map(
    lambda x, y: preprocess(x, y, False),
    num_parallel_calls=AUTOTUNE
)

# 🚀 CACHE AFTER MAP (IMPORTANT)
train_ds = train_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ---------------- MODEL ----------------
base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)

# ⚡ Freeze more layers for speed
for layer in base.layers[:-20]:
    layer.trainable = False
for layer in base.layers[-20:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.35)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5)
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------- SAVE ----------------
model.save(MODEL_PATH)
print("\n✅ MODEL SAVED:", MODEL_PATH)
