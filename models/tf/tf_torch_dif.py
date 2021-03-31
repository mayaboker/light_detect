from models.common import ConvBn, DwConvBn
from models.FpnHead import UpSample

from models.tf.common import ConvBn as ConvBnTf
from models.tf.common import DwConvBn as DwConvBnTf
from models.tf.FpnHead import UpSample as UpSampleTf

import numpy as np
import torch
import tensorflow as tf


# TEST ConvBN
batch = 1
in_c = 64
h = 320
w = 160
out_c = 64
stride = 2
kernel = 3

x = np.random.rand(batch, in_c, h, w).astype(np.float32)
# x = np.ones([batch, in_c, h, w]).astype(np.float32)

x_tf = tf.convert_to_tensor(np.transpose(x, [0, 2, 3, 1]))
# print('x tf', x_tf.shape)
x_torch = torch.from_numpy(x)
# print('x_torch', x_torch.shape)

weights = np.random.rand(out_c, in_c, kernel, kernel).astype(np.float32)
# weights = np.ones([out_c, in_c, kernel, kernel]).astype(np.float32)

conv_torch = ConvBn(in_c, out_c=out_c, k=kernel, s=stride).eval()
conv_torch.conv[0].weight.data = torch.from_numpy(weights)

conv_tf = ConvBnTf(filters=out_c, kernel_size=kernel, strides=stride, use_bias=False, weights=[np.transpose(weights, [2, 3, 1, 0])])

y_tf = conv_tf(x_tf)
y_tf = np.transpose(y_tf.numpy(), [0, 3, 1, 2])
# print('y tf', y_tf.shape)
y_torch = conv_torch(x_torch)
y_torch = y_torch.detach().numpy()
# print('y_torch', y_torch.shape)

np.testing.assert_allclose(y_torch, y_tf, rtol=1e-4, atol=1e-5)
print('ConvBn Good!')



# TEST DwConvBN
batch = 2
in_c = 64
h = 320
w = 160
out_c = 64
stride = 2
kernel = 3

x = np.random.rand(batch, in_c, h, w).astype(np.float32)
# x = np.ones([batch, in_c, h, w]).astype(np.float32)

x_tf = tf.convert_to_tensor(np.transpose(x, [0, 2, 3, 1]))
# print('x tf', x_tf.shape)
x_torch = torch.from_numpy(x)
# print('x_torch', x_torch.shape)

weights = np.random.rand(in_c, 1, kernel, kernel).astype(np.float32)
# weights = np.ones([out_c, in_c, kernel, kernel]).astype(np.float32)

conv_torch = DwConvBn(in_c, k=kernel, s=stride).eval()
conv_torch.conv[0].weight.data = torch.from_numpy(weights)

conv_tf = DwConvBnTf(kernel_size=kernel, strides=stride, use_bias=False, weights=[np.transpose(weights, [2, 3, 0, 1])])

y_tf = conv_tf(x_tf)
y_tf = np.transpose(y_tf.numpy(), [0, 3, 1, 2])
# print('y tf', y_tf.shape)
y_torch = conv_torch(x_torch)
y_torch = y_torch.detach().numpy()
# print('y_torch', y_torch.shape)

np.testing.assert_allclose(y_torch, y_tf, rtol=1e-4, atol=1e-5)
print('DwConvBn Good!')


# TEST Upsample
batch = 1
in_c = 24
h = 40
w = 40

x = np.random.rand(batch, in_c, h, w).astype(np.float32)
# x = np.ones([batch, in_c, h, w]).astype(np.float32)

x_tf = tf.convert_to_tensor(np.transpose(x, [0, 2, 3, 1]))
# print('x tf', x_tf.shape)
x_torch = torch.from_numpy(x)
# print('x_torch', x_torch.shape)

weights = np.random.rand(in_c, in_c, 1, 1).astype(np.float32)
# weights = np.ones([out_c, in_c, kernel, kernel]).astype(np.float32)

conv_torch = UpSample(in_c).eval()
conv_torch.up[1].conv[0].weight.data = torch.from_numpy(weights)

conv_tf = UpSampleTf(in_c)
conv_tf.build([batch, h, w, in_c])
conv_tf.layers[1].layers[1].set_weights([np.transpose(weights, [2, 3, 1, 0])])

y_tf = conv_tf(x_tf)
y_tf = np.transpose(y_tf.numpy(), [0, 3, 1, 2])
# print('y tf', y_tf.shape)
y_torch = conv_torch(x_torch)
y_torch = y_torch.detach().numpy()
# print('y_torch', y_torch.shape)

np.testing.assert_allclose(y_torch, y_tf, rtol=1e-4, atol=1e-5)

print('Upsample Good!')



