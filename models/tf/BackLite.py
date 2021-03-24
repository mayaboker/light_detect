import tensorflow as tf
from models.tf.common import ConvBn, DwConvBn


class Block(tf.keras.Model):
    def __init__(self, out_c, s, res=False):
        super(Block, self).__init__()
        self.res = res
        self.block = tf.keras.Sequential([
            ConvBn(filters=out_c, kernel_size=1, strides=1),
            DwConvBn(kernel_size=3, strides=s)
        ])

    def call(self, x):
        y = self.block(x)
        if self.res:
            y = y + x
        return y


class BigBlock(tf.keras.Model):
    def __init__(self, out_c, n, s):
        super(BigBlock, self).__init__()
        model = []
        for i in range(n):
            if i == 0:
                block = Block(out_c, s, res=False)
            else:
                block = Block(out_c, 1, res=True)
            model.append(block)

        self.big_block = tf.keras.Sequential(*model)
    
    def forward(self, x):
        return self.big_block(x)


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

        self.stem = ConvBn(filters=in_planes, kernel_size=3, strides=2)

        self.blocks = []
        for settings in self.down_settings:
            self.blocks.append(
                BigBlock(settings[0], n=settings[1], s=settings[2])
            )

        self.head_convs = [
            ConvBn(filters=head_ch, kernel_size=1, strides=1) for _ in self.feat_channels
        ]        

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.down_settings[i][3]:
                feats.insert(0, 
                    self.head_convs[len(feats)](x)
                )
        return feats