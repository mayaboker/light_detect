import tensorflow as tf
from models.tf.common import ConvBn, DwConvBn


def UpSample(in_c, mode='interpolate'):
    if mode == 'conv':
        # TODO - support tf conv transpose
        model = None
    elif mode == 'interpolate':
        model = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D(size=2, interpolation='nearest'),
            ConvBn(filters=in_c, kernel_size=1, strides=1, use_bias=False)
        ])

    return model


class FpnHead(tf.keras.Model):
    def __init__(self, head_ch, num_laterals, upsample_mode='interpolate', one_feat_map=True):
        super(FpnHead, self).__init__()
        
        self.ups = [
            UpSample(head_ch, mode=upsample_mode)
            for _ in range(num_laterals-1)
        ]
        self.one_feat_map = one_feat_map
        self.num_outputs = 1 if one_feat_map else num_laterals
        self.conv_feats = [
            ConvBn(filters=head_ch, kernel_size=3, strides=1, use_bias=False)
            for _ in range(self.num_outputs)
        ]

    def call(self, feats):
        x = feats[0]
        outs = []
        for i in range(1, len(feats)):
            if not self.one_feat_map:
                outs.append(
                    self.conv_feats[i-1](x))
            x = self.ups[i-1](x) + feats[i]
        outs.append(
            self.conv_feats[-1](x))
        return outs

if __name__ == "__main__":
    input_shapes = [(1, 10, 10, 24), (1, 20, 20, 24), (1, 40, 40, 24), (1, 80, 80, 24)] 
    x = []
    for shape in input_shapes:
        x.append(tf.random.normal(shape))

    model = FpnHead(24, 4)
    y = model(x)
    for _y in y:
        print(_y.shape)