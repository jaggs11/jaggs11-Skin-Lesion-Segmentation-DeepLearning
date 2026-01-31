import tensorflow as tf
from data_processing import get_pairs
from data_augmentation import make_dataset
from unet import build_unet
from attention_unet import build_attention_unet
from xception_unet import build_xception_unet
from utils import dice_coef, bce_dice, accuracy, plot_history

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
IMAGE_DIR = "C:/Users/Nayan Jaggi/Desktop/ai/ISIC2018/images"
MASK_DIR  = "C:/Users/Nayan Jaggi/Desktop/ai/ISIC2018/masks"

# -------------------------------------------------------
# LOAD ALL IMAGE–MASK PAIRS
# -------------------------------------------------------
pairs = get_pairs(IMAGE_DIR, MASK_DIR)
total = len(pairs)

train_size = int(0.8 * total)
test_size  = total - train_size

train_pairs = pairs[:train_size]
test_pairs  = pairs[train_size:]

print("TOTAL PAIRS FOUND:", total)
print("TRAIN:", len(train_pairs))
print("TEST:", len(test_pairs))

# -------------------------------------------------------
# CREATE DATASETS
# -------------------------------------------------------
train_ds = make_dataset(train_pairs, batch_size=2, training=True)
test_ds  = make_dataset(test_pairs, batch_size=2, training=False)

# -------------------------------------------------------
# TRAIN FUNCTION (NO VALIDATION)
# -------------------------------------------------------
def train(model, optimizer_name, lr, epochs=20):
    if optimizer_name == "adam":
        opt = tf.keras.optimizers.Adam(lr)
    else:
        opt = tf.keras.optimizers.SGD(lr, momentum=0.9)

    model.compile(
        optimizer=opt,
        loss=bce_dice,
        metrics=[dice_coef, accuracy]
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=20   # speed boost
    )

    return history

# -------------------------------------------------------
# TRAIN ALL THREE MODELS
# -------------------------------------------------------

print("\n================ TRAINING U-Net =================")
unet = build_unet()
hist_unet = train(unet, "adam", 1e-4)

print("\n================ TRAINING Attention U-Net =================")
att = build_attention_unet()
hist_att = train(att, "adam", 1e-4)

print("\n================ TRAINING Xception-U-Net Hybrid =================")
xunet = build_xception_unet()
hist_xunet = train(xunet, "adam", 1e-4)

# -------------------------------------------------------
# PLOT TRAINING GRAPHS
# -------------------------------------------------------

plot_history(hist_unet, "UNet")
plot_history(hist_att, "Attention UNet")
plot_history(hist_xunet, "Xception UNet")

# -------------------------------------------------------
# TEST SET PERFORMANCE
# -------------------------------------------------------

print("\n================ TEST SET EVALUATION ================")

print("\nU-Net TEST results: (loss, dice, accuracy)")
print(unet.evaluate(test_ds))

print("\nAttention U-Net TEST results: (loss, dice, accuracy)")
print(att.evaluate(test_ds))

print("\nXception-U-Net TEST results: (loss, dice, accuracy)")
print(xunet.evaluate(test_ds))

# -------------------------------------------------------
# FINAL ACCURACY COMPARISON
# -------------------------------------------------------
print("\n================ FINAL ACCURACY COMPARISON ================")
print("U-Net Train Accuracy:", hist_unet.history['binary_accuracy'][-1])
print("Attention U-Net Train Accuracy:", hist_att.history['binary_accuracy'][-1])
print("Xception U-Net Train Accuracy:", hist_xunet.history['binary_accuracy'][-1])
