import tensorflow as tf
import os
from glob import glob

# -----------------------------
# GLOBAL IMAGE SIZE
# -----------------------------
IMG_SIZE = 224   # Change to 256 later if needed


# -----------------------------
# LOAD & PREPROCESS ONE IMAGE–MASK PAIR
# -----------------------------
def load_image(img_path, mask_path):
    # Load image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0   # normalize 0–1

    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")

    # Convert to binary mask (0 or 1)
    mask = tf.cast(mask > 127, tf.float32)

    return img, mask


# -----------------------------
# CREATE IMAGE–MASK PAIRS FROM FOLDERS
# -----------------------------
def get_pairs(IMAGE_DIR, MASK_DIR):
    # Collect all images and masks
    images = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))
    masks  = sorted(glob(os.path.join(MASK_DIR, "*_segmentation.png")))

    pairs = []

    for img in images:
        base = os.path.basename(img).replace(".jpg", "")
        mask_path = os.path.join(MASK_DIR, base + "_segmentation.png")

        # Only add if mask actually exists (some ISIC images have no mask)
        if os.path.exists(mask_path):
            pairs.append((img, mask_path))

    print(f"✔ Total usable image–mask pairs: {len(pairs)}")

    return pairs
