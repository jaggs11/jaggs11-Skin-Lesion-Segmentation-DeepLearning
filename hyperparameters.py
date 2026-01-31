from unet import build_unet
from attention_unet import build_attention_unet
from xception_unet import build_xception_unet
from training import train
from data_processing import get_pairs
from data_augmentation import make_dataset

IMAGE_DIR = "C:/Users/Nayan Jaggi/Desktop/AI/ISIC2018/images"
MASK_DIR  = "C:/Users/Nayan Jaggi/Desktop/AI/ISIC2018/masks"


pairs = get_pairs(IMAGE_DIR, MASK_DIR)

train_pairs = pairs[:600]
val_pairs   = pairs[600:700]

train_ds = make_dataset(train_pairs, 8, True)
val_ds   = make_dataset(val_pairs, 8, False)

configs = [
    ("UNet", "adam", 1e-4),
    ("UNet", "sgd", 1e-3),

    ("AttentionUNet", "adam", 1e-4),
    ("AttentionUNet", "sgd", 1e-3),

    ("XceptionUNet", "adam", 1e-4),   # NEW
    ("XceptionUNet", "sgd",  1e-3)    # NEW
]

for model_name, opt, lr in configs:
    print(f"\n🔵 Running {model_name} with {opt} LR={lr}")

    if model_name == "UNet":
        model = build_unet()
    elif model_name == "AttentionUNet":
        model = build_attention_unet()
    else:
        model = build_xception_unet()  # NEW HYBRID

    train(model, opt, lr, epochs=20)
