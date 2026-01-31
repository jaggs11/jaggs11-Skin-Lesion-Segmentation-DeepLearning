import tensorflow as tf

from tensorflow.keras import layers, Model

from tensorflow.keras.layers import LeakyReLU

def conv_block(x, filters):

    kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)


    # 1) ReLU activation  --------------------
    x = layers.Conv2D(
    filters, 
    3, 
    padding="same",
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-5)
    )(x)


    # 2) LeakyReLU activation  ---------------
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # 3) Tanh activation ---------------------
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.Activation("tanh")(x)

    return x


def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling2D()(c)
    return c, p

def decoder_block(x, skip, filters):
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(img_size=224):
    inputs = layers.Input((img_size, img_size, 3))

    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)

    bn = conv_block(p4, 1024)

    d4 = decoder_block(bn, c4, 512)
    d3 = decoder_block(d4, c3, 256)
    d2 = decoder_block(d3, c2, 128)
    d1 = decoder_block(d2, c1, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return Model(inputs, outputs)
