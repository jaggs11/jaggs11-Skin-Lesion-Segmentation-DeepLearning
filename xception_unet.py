import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.applications import Xception


def decoder_block(x, skip, filters):

    # Upsample
    x = layers.UpSampling2D((2, 2))(x)

    # Resize skip connection to match
    x_h, x_w = x.shape[1], x.shape[2]
    skip = layers.Resizing(x_h, x_w)(skip)

    # Concatenate
    x = layers.Concatenate()([x, skip])

    # REGULARIZATION (L1 + L2)
    reg = tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)

    # ------------------------------
    # 1) Conv + ReLU
    # ------------------------------
    x = layers.Conv2D(
        filters, 3, padding="same",
        activation="relu",
        kernel_regularizer=reg
    )(x)

    # ------------------------------
    # 2) Conv + LeakyReLU
    # ------------------------------
    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_regularizer=reg
    )(x)
    x = LeakyReLU(alpha=0.1)(x)

    # ------------------------------
    # 3) Conv + Tanh
    # ------------------------------
    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_regularizer=reg
    )(x)
    x = layers.Activation("tanh")(x)

    return x



# ----------------------------
#  Xception-U-Net Hybrid
# ----------------------------
def build_xception_unet(input_shape=(224, 224, 3)):
    inputs = layers.Input(input_shape)

    # ----------------------------
    # ENCODER (Pretrained Xception)
    # ----------------------------
    base = Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    # Extract skip connections (Xception uses uneven downsample sizes)
    s1 = base.get_layer("block1_conv2_bn").output   # ~112×112
    s2 = base.get_layer("block3_sepconv2_bn").output # ~56×56
    s3 = base.get_layer("block4_sepconv2_bn").output # ~28×28
    s4 = base.get_layer("block13_sepconv2_bn").output # ~14×14

    # Bottleneck
    bn = base.get_layer("block14_sepconv2_act").output  # ~7×7×2048

    # ----------------------------
    # DECODER (U-Net style)
    # ----------------------------
    d1 = decoder_block(bn, s4, 512)   # 7 → 14
    d2 = decoder_block(d1, s3, 256)   # 14 → 28
    d3 = decoder_block(d2, s2, 128)   # 28 → 56
    d4 = decoder_block(d3, s1, 64)    # 56 → 112

    # Last upsample to full 224×224
    d5 = layers.UpSampling2D((2, 2))(d4)
    d5 = layers.Conv2D(32, 3, activation="relu", padding="same")(d5)

    # Output mask
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d5)

    model = Model(inputs, outputs)
    return model
