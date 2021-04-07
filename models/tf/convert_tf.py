import json
import argparse
import sys
import torch
import tensorflow as tf
import numpy as np
from utils.utils import load_yaml
from factory import get_fpn_net


def load_module_weight(torch_ckpt, tf_model, mapping_table, layer=None, name=''):
    for l in layer.layers:
        n = name if name == '' else name + '.' 
        n += l.name
        if hasattr(l, 'layers'):
            load_module_weight(torch_ckpt, tf_model, mapping_table, layer=l, name=n)
        else:
            if n not in mapping_table:
                print(f'Ignore layer: {l.name}')
                continue
            print(f'Set layer: {n}')
            layer_type = n.split('.')[-1]
            torch_layer_names = mapping_table[n]

            if layer_type == 'conv2d':
                weights = []
                weight = np.array(torch_ckpt[torch_layer_names[0]])

                if n.split('.')[-2] == 'dw_conv_bn': # TODO add specific case after changing DW layer name
                    weight = np.transpose(weight, [2, 3, 0, 1])
                else: # regular conv2d
                    weight = np.transpose(weight, [2, 3, 1, 0])

                weights.append(weight)
                if len(torch_layer_names) == 2: # if have bias
                    bias = np.array(torch_ckpt[torch_layer_names[1]])
                    weights.append(bias)
                l.set_weights(weights)

            elif layer_type == 'batch_norm':
                l_name = torch_layer_names[0]
                gamma = np.array(torch_ckpt[l_name + '.weight'])
                beta = np.array(torch_ckpt[l_name + '.bias'])
                running_mean = np.array(torch_ckpt[l_name + '.running_mean'])
                running_var = np.array(torch_ckpt[l_name + '.running_var'])
                l.set_weights([gamma, beta, running_mean, running_var])
            
            else:
                raise RuntimeError(f'Unknown Layer type \'{layer_type}\'.')



def load_weight(torch_ckpt, tf_model, mapping_table):
    mapping_table = {layer['name']: layer['weights'] for layer in mapping_table.values()}
    load_module_weight(torch_ckpt, tf_model, mapping_table, layer=tf_model)


def tf_model_checker(torch_ckpt, tf_model):
    cfg = load_yaml('config.yaml')
    torch_model = get_fpn_net(cfg['net'])
    torch_model.load_state_dict(torch_ckpt)
    torch_model.eval()

    x = np.random.rand(2, 3, 320, 320).astype(np.float32)
    x_tf = tf.convert_to_tensor(np.transpose(x, [0, 2, 3, 1]))
    x_torch = torch.from_numpy(x)

    y_tf = tf_model(x_tf)
    y_torch = torch_model(x_torch)

    for f, c in zip(y_tf[0], y_torch[0]):
        f = np.transpose(f.numpy(), [0, 3, 1, 2])
        c = c.detach().numpy()
        np.testing.assert_allclose(c, f, rtol=1e-4, atol=1e-5)
    
    print('All Good!')

    
def create_mapping_template(tf_model, json_path):
    mapping = print_modules_names(layer=tf_model)
    with open('mappin_table.json', 'w') as j:
        json.dump(mapping, j, indent=4)

def print_modules_names(layer=None, name=''):
    mapping = {}
    for l in layer.layers:
        n = name if name == '' else name + '.' 
        n += l.name
        if hasattr(l, 'layers'):
            mapping.update(print_modules_names(layer=l, name=n))
        else:
            layer_map = {}
            weights = l.get_weights()
            if len(weights) > 0:
                shapes = [x.shape for x in weights]
                layer_map['name'] = n
                layer_map['shapes'] = shapes
                layer_map['weights'] = []
                mapping[n] = layer_map
    return mapping


def convert(torch_ckpt, tf_model, mapping_table, models_path, model_name):
    load_weight(torch_ckpt, tf_model, mapping_table)
    
    tf_model_checker(torch_ckpt, tf_model)

    tf_model.compute_output_shape(input_shape=(None, 320,320,3))
    tf_model.save(f'{models_path}/{model_name}/', include_optimizer=False)
    convert_tflite(tf_model, models_path, model_name)
    print("Converted to tflite")


def convert_tflite(tf_model, models_path, model_name):
    # Convert the model to the TensorFlow Lite format without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_tflite = converter.convert()
    # # Save the model to disk
    with open(f'{models_path}/{model_name}/{model_name}.tflite', "wb") as f:
        f.write(model_tflite)
    
    # Convert the model to the TensorFlow Lite format with quantization
    # TODO - add representive dataset
    def representative_dataset():
        for i in range(160):
            input_shape = (320, 320, 3)
            x = tf.random.normal(input_shape)
            yield [np.expand_dims(x, axis=0)]

    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce integer only quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = representative_dataset
    model_quant = converter.convert()

    # Save the model to disk
    with open(f'{models_path}/{model_name}/{model_name}_quant.tflite', "wb") as f:
        f.write(model_quant)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='convert model')

    parser.add_argument('--ckpt', type=str, default='../tf_convert/sample_torch.pth', help='torch checkpoint path')
    parser.add_argument('--name', type=str, default='test_tf', help='name of output model')
    parser.add_argument('--map', type=str, default='./models/tf/mapping_table.json', help='Mapping table json file')
    args = parser.parse_args()
    
    cfg = load_yaml('config.yaml')
    models_path = cfg['paths']['converted_models']
    
    model_name = args.name
    if model_name is None:
        raise Exception('Must enter a name, --name')

    torch_ckpt = args.ckpt
    torch_ckpt = torch.load(torch_ckpt, map_location='cpu')['net_state_dict']

    with open(args.map, 'r') as j:
        mapping_table = json.load(j) 
    

    tf_model = get_fpn_net(cfg['net'], framework='tf')
    tf_model.build((1, 320, 320, 3))

    convert(torch_ckpt, tf_model, mapping_table, models_path, model_name)
