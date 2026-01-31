import tensorflow as tf
from data_processing import get_pairs
from data_augmentation import make_dataset
from unet import build_unet
from utils import dice_coef, accuracy, bce_dice
import pandas as pd

IMAGE_DIR = "C:/Users/Nayan Jaggi/Desktop/ai/ISIC2018/images"
MASK_DIR  = "C:/Users/Nayan Jaggi/Desktop/ai/ISIC2018/masks"

pairs = get_pairs(IMAGE_DIR, MASK_DIR)
train_pairs = pairs[:800]
val_pairs   = pairs[800:900]

def run_experiment(lr, optimizer, batch):
    print(f"\n🔹 Testing LR={lr}, OPT={optimizer}, BATCH={batch}")

    train_ds = make_dataset(train_pairs, batch_size=batch, training=True)
    val_ds   = make_dataset(val_pairs, batch_size=batch, training=False)

    model = build_unet()

    if optimizer == "adam":
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
        validation_data=val_ds,
        epochs=3,              # small, fast tuning
        steps_per_epoch=20,
        validation_steps=5,
        verbose=1
    )

    final_dice = history.history["val_dice_coef"][-1]
    return final_dice


# ------------------------------------------
#  Hyperparameter Search
# ------------------------------------------

results = []

learning_rates = [1e-3, 1e-4, 1e-5]
optimizers     = ["adam", "sgd"]
batch_sizes    = [2, 4]

for lr in learning_rates:
    for opt in optimizers:
        for b in batch_sizes:
            dice = run_experiment(lr, opt, b)
            results.append([lr, opt, b, dice])

df = pd.DataFrame(results, columns=["LR", "Optimizer", "Batch", "Val Dice"])
df.to_csv("tuning_results.csv", index=False)
print("\n📌 TUNING COMPLETE! Results saved to tuning_results.csv")
print(df)
