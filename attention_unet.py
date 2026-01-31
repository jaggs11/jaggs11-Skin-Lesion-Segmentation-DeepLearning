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



def attention_gate(x, g, filters):
    """
    x = skip connection (same spatial size as g)
    g = gating signal
    """

    theta_x = layers.Conv2D(filters, 1, padding="same")(x)
    phi_g   = layers.Conv2D(filters, 1, padding="same")(g)

    add     = layers.Add()([theta_x, phi_g])
    act     = layers.Activation("relu")(add)

    psi     = layers.Conv2D(1, 1, activation="sigmoid")(act)

    return layers.Multiply()([x, psi])


def decoder_block_att(x, skip, filters):
    g = layers.UpSampling2D((2, 2))(x)         # make decoder output same size as skip
    att = attention_gate(skip, g, filters)     # apply attention after match

    concat = layers.Concatenate()([g, att])
    return conv_block(concat, filters)


def build_attention_unet(input_shape=(224, 224, 3)):
    inputs = layers.Input(input_shape)

    # ----- Encoder -----
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    bn = conv_block(p4, 1024)

    # ----- Decoder -----
    d4 = decoder_block_att(bn, c4, 512)
    d3 = decoder_block_att(d4, c3, 256)
    d2 = decoder_block_att(d3, c2, 128)
    d1 = decoder_block_att(d2, c1, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)

    return Model(inputs, outputs)
