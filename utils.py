import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------
# METRICS
# -------------------

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2*inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

bce = tf.keras.losses.BinaryCrossentropy()

def bce_dice(y_true, y_pred):
    return bce(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))

# NEW METRIC:
accuracy = tf.keras.metrics.BinaryAccuracy()


# -------------------
# PLOTTING
# -------------------
import matplotlib.pyplot as plt

def plot_history(history, title=""):
    h = history.history

    plt.figure(figsize=(10, 4))

    # ----------------- LOSS -----------------
    plt.subplot(1, 2, 1)
    plt.plot(h["loss"], label="train_loss")
    if "val_loss" in h:
        plt.plot(h["val_loss"], label="val_loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ----------------- DICE / ACC -----------------
    plt.subplot(1, 2, 2)

    # Dice
    if "dice_coef" in h:
        plt.plot(h["dice_coef"], label="train_dice")
    if "val_dice_coef" in h:
        plt.plot(h["val_dice_coef"], label="val_dice")

    # Accuracy
    if "binary_accuracy" in h:
        plt.plot(h["binary_accuracy"], label="train_acc")
    if "val_binary_accuracy" in h:
        plt.plot(h["val_binary_accuracy"], label="val_acc")

    plt.title(f"{title} - Dice / Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()


# -------------------
# VISUALIZE PREDICTIONS
# -------------------
def show_predictions(model, dataset, n=3):
    imgs, masks = next(iter(dataset))
    preds = model.predict(imgs)

    for i in range(n):
        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1); plt.imshow(imgs[i]); plt.title("Image")
        plt.subplot(1,3,2); plt.imshow(masks[i], cmap='gray'); plt.title("Mask")
        plt.subplot(1,3,3); plt.imshow(preds[i] > 0.5, cmap='gray'); plt.title("Prediction")
        plt.show()
