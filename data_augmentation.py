import tensorflow as tf
from data_processing import load_image

def augment(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.adjust_brightness(img, 0.1)
    return img, mask


def make_dataset(pairs, batch_size=8, training=True):
    img_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    def load_and_aug(img_path, mask_path):
        img, mask = tf.py_function(
            func=load_image,
            inp=[img_path, mask_path],
            Tout=[tf.float32, tf.float32]
        )

        img.set_shape([224, 224, 3])
        mask.set_shape([224, 224, 1])

        if training:
            img, mask = augment(img, mask)
        return img, mask

    ds = ds.map(load_and_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
