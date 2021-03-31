import tensorflow as tf


def ConvBn(name='conv_bn', *args, **kwargs):
    pad = kwargs['kernel_size'] // 2
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(pad, name='pad'),
        tf.keras.layers.Conv2D(*args, **kwargs, name='conv2d'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5, name='batch_norm'),
        tf.keras.layers.ReLU(name='relu'),
    ], name=name)


def DwConvBn(*args, **kwargs):
    pad = kwargs['kernel_size'] // 2
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(pad, name='pad'),
        tf.keras.layers.DepthwiseConv2D(*args, **kwargs, name='conv2d'), # TODO replace name to dw_conv2d
        tf.keras.layers.BatchNormalization(epsilon=1e-5, name='batch_norm'),
        tf.keras.layers.ReLU(name='relu'),
    ], name='dw_conv_bn')


if __name__ == "__main__":
    input_shape = (2, 32, 32, 3)
    x = tf.random.normal(input_shape)

    
    conv = ConvBn(filters=64, kernel_size=3, strides=2)
    y = conv(x)
    print(y.shape)

