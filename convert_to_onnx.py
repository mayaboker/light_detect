import os.path as osp
from pathlib import Path
import argparse
import torch
import onnx
from models.common import MaxPoolNms
from utils.utils import load_yaml
from factory import get_fpn_net


def multiple_names(out_names, torch_out, in_dims):
    new_out_names = []
    for outs in torch_out:
        stride = in_dims[2] // outs[0].size(2)
        new_out_names += [f'{name}_s{stride}' for name in out_names]
    return out_names


def create_dynamic_axes_dict(in_names, out_names):
    names = in_names + out_names
    d = {}
    for name in names:
        d[name] = {0: 'batch'}
    return d


def convert_to_onnx(model, path, onnx_name, in_dims=(1, 3, 800, 800), out_names=None, in_names=None, one_feat=False):
    if out_names is None:
        out_names = ['heatmap', 'scale', 'offset']
    if in_names is None:
        in_names = ['output']

    path = f'{path}/{onnx_name}'
    print(f'Saving model to {path}')
    Path(path).mkdir(exist_ok=True, parents=True)
    model_path = osp.join(path, f'{onnx_name}.onnx')

    model.eval()

    x = torch.randn(*in_dims)
    torch_out = model(x)
    
    if not one_feat:
        out_names = multiple_names(out_names, torch_out, in_dims)
    
    dynamic_axes = create_dynamic_axes_dict(in_names, out_names)

    torch.onnx.export(
        model,
        x,
        model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes
    )

    print(f'saved onnx models: {model_path}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help='Output model name')

    args = parser.parse_args()

    cfg = load_yaml('config.yaml')
    models_path = cfg['paths']['converted_models']
    one_feat = cfg['net']['one_feat_map']
    
    model_name = args.name
    if model_name is None:
        raise Exception('Must enter a name, --name')


    net = get_fpn_net(cfg['net'])
    net.eval()
    convert_to_onnx(net, models_path, model_name, one_feat=one_feat)

