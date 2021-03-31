import tensorflow as tf
from models.tf.common import ConvBn, DwConvBn


class Block(tf.keras.Model):
    def __init__(self, out_c, s, res=False, name='block'):
        super(Block, self).__init__()
        self._name = name
        self.res = res
        
        self.conv_bn = ConvBn(filters=out_c, kernel_size=1, strides=1, use_bias=False)
        self.dw_conv_bn = DwConvBn(kernel_size=3, strides=s, use_bias=False)

    def call(self, x):
        # y = self.block(x)
        y = self.conv_bn(x)
        y = self.dw_conv_bn(y)
        if self.res:
            y = y + x
        return y


def BigBlock(out_c, n, s, name='bigblock'):
    model = []
    for i in range(n):
        if i == 0:
            block = Block(out_c, s, res=False, name=f'block_{i}')
        else:
            block = Block(out_c, 1, res=True, name=f'block_{i}')
        model.append(block)

    return tf.keras.Sequential([*model], name=name)
    
    # def call(self, x):
    #     return self.big_block(x)


class Backbone(tf.keras.Model):
    def __init__(self, head_ch):
        super(Backbone, self).__init__()

        self.down_settings = [
            # c  n  s   f
            [16, 1, 1, False],
            [24, 2, 2, True],
            [32, 3, 2, True],
            [64, 2, 1, False],
            [96, 4, 2, True],
            [160, 2, 2, False],
            [320, 1, 1, True],
        ]
        self.feat_channels = [24, 32, 96, 320]
        in_planes = 64

        self.stem = ConvBn(name='stem', filters=in_planes, kernel_size=3, strides=2, use_bias=False)

        self.blocks = []
        for i, settings in enumerate(self.down_settings):
            self.blocks.append(
                BigBlock(settings[0], n=settings[1], s=settings[2], name=f'bigblock_{i}')
            )

        self.head_convs = [
            ConvBn(filters=head_ch, kernel_size=1, strides=1, name=f'head_convs_{i}', use_bias=False)
            for i, _ in enumerate(self.feat_channels)
        ]        

    def call(self, x):
        x = self.stem(x)
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.down_settings[i][3]:
                feats.insert(0, 
                    self.head_convs[len(feats)](x)
                )
        return feats

if __name__ == "__main__":
    input_shape = (1, 320, 320, 3)
    x = tf.random.normal(input_shape)

    model = Backbone(24)
    y = model(x)
    for _y in y:
        print(_y.shape)