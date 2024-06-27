import tensorflow as tf
from tensorflow.keras import layers




class Block(tf.keras.Model):
    """ConvNeXt block.

    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.dim = dim
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None
        self.dw_conv_1 = layers.Conv2D(
            filters=dim, kernel_size=7, padding="same", groups=dim
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.pw_conv_1 = layers.Dense(4 * dim)
        self.act_fn = layers.Activation("gelu")
        self.pw_conv_2 = layers.Dense(dim)


    def call(self, inputs):
        x = inputs

        x = self.dw_conv_1(x)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = self.act_fn(x)
        x = self.pw_conv_2(x)
        x = ECALayer()(x)


        return inputs + x



def DownsampleLayer(dim):

    downsample_layer = tf.keras.Sequential([
        layers.LayerNormalization(epsilon=1e-6),
        layers.Conv2D(dim, kernel_size=2, strides=2)
    ])

    return downsample_layer



class ECALayer(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = tf.keras.Sequential([
            layers.Dense(self.channels // self.reduction_ratio, activation='relu'),
            layers.Dense(self.channels, activation='sigmoid')
        ])
        super(ECALayer, self).build(input_shape)

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        weights = self.fc(avg_pool)
        weights = tf.reshape(weights, [-1, 1, 1, self.channels])
        return inputs * weights




