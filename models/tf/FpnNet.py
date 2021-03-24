import tensorflow as tf
from models.tf.FpnHead import FpnHead


class FpnNet(tf.keras.Model):
    def __init__(self, backbone, head_ch, reg_dict, upsample_mode='interpolate', one_feat_map=True):
        super(FpnNet, self).__init__()
        self.head_ch = head_ch
        self.reg_dict = reg_dict

        self.base = backbone
        num_laterals = len(backbone.feat_channels)
        self.fpn = FpnHead(head_ch, num_laterals, upsample_mode=upsample_mode, one_feat_map=one_feat_map)
        
        self.num_maps = 1 if one_feat_map else num_laterals
        self.reg_heads = [
            self.build_regression_head() for _ in range(self.num_maps)
        ]
       
            
    def build_regression_head(self):
        reg_head_dict = {}
        for head, num_channels in self.reg_dict.items():
            m_head = tf.keras.Sequential()
            activation = None
            if head == 'hm':
                activation = 'sigmoid'
            m_head.add(
                tf.keras.layers.Conv2D(filters=num_channels, kernel_size=1, strides=1, use_bias=True, activation=activation)
            )            
            reg_head_dict[head] = m_head

        return reg_head_dict
            

    def call(self, x):
        feats = self.base(x)
        p_feats = self.fpn(feats)
        outs = []
        for i, p_feat in enumerate(p_feats):
            reg_outs = []
            for reg_head in self.reg_dict.keys():
                reg_outs.append(
                    self.reg_heads[i][reg_head](p_feat)
                )
            outs.append(reg_outs)

        return outs


if __name__ == "__main__":
    from models.tf.BackLite import Backbone
    head_ch = 24
    reg_dict = {'hm': 1, 'of': 2, 'wh': 2}
    backbone = Backbone(head_ch)
    net = FpnNet(backbone, head_ch, reg_dict, one_feat_map=True)

    input_shape = (1, 320, 320, 3)
    x = tf.random.normal(input_shape)
    
    y = net(x)
    for e in y:
        print('==================')
        for j in e:
            print(j.shape)