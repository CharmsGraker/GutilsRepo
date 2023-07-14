import base64

import PIL

import io
import os.path
from importlib import import_module

import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig

from Gutils.annot import AutoInjectConfigParams, StrictType
from Gutils.config import GDict


def find_class(location: str):
    ret = location.split('.')
    package = ".".join(ret[:-1])
    method = ret[-1]
    module = import_module(package)
    return getattr(module, method)


def base642img(imgbase64):
    img = io.BytesIO(base64.b64decode(imgbase64.encode('utf-8')))
    img = PIL.Image.open(img)
    return img


def create_instance_from_config(config: DictConfig):
    kind_key = 'kind' if 'kind' in config else 'target'
    T = config[kind_key]

    return find_class(T)(**config['params'])


def create_instance(location: str, init_params: dict):
    return find_class(location)(**init_params)


def mkdir_if_not_exists(dir):
    if len(dir) == 0:
        return
    if os.path.exists(dir):
        return
    os.makedirs(dir, False)
    os.chmod(dir, 0o777)


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


@AutoInjectConfigParams()
def save_ckpt(ckpt_name, models, optimizers=None, n_iter=0):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            ckpt_dict[prefix] = optimizer.state_dict()
    mkdir_if_not_exists(os.path.dirname(ckpt_name))
    torch.save(ckpt_dict, ckpt_name)


@AutoInjectConfigParams(suppressRedundantParams=True)
def load_checkpoint(ckpt_name: str, model: nn.Module, map_location="cuda", name_mapping=None, model_key=None,
                    strict=True):
    if os.path.isdir(ckpt_name):
        # 如果提供的是文件夹，读取排序最大的一个，一般来说都是step最大的那个
        ckpt_files = os.listdir(ckpt_name)
        assert ckpt_files is not None
        ckpt_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        ckpt_name = os.path.join(ckpt_name, ckpt_files[-1])

    ckpt_dict = torch.load(ckpt_name, map_location=map_location)
    if model_key is None:
        state_dict = ckpt_dict
    else:
        if model_key not in ckpt_dict:
            raise ValueError(
                f"error occurred when loading checkpoint path at: {ckpt_name}\ncheckpoint dict has only keys below: {ckpt_dict.keys()}")
        state_dict = ckpt_dict[model_key]

    if not isinstance(name_mapping, dict):
        name_mapping = {}
    if len(name_mapping) > 0:
        # rename parameters in ckpt, any else key not startswith prefix expected will be ignored
        new_ckpt_dict = {}
        for k, v in state_dict.items():
            need_replace = False
            for old_pre, new_pre in name_mapping.items():
                if k.startswith(old_pre):
                    new_k = new_pre + k[len(old_pre):]
                    new_ckpt_dict[new_k] = v
                    need_replace = True
                    break
            # TODO: 可以只是更新state_dict，而不是只加载name_mapping中的state_dict
        state_dict = new_ckpt_dict
    model.load_state_dict(state_dict, strict=strict)

    print(f"[*] LOAD SUCCESS, model at path {ckpt_name} has loaded.")
    if 'n_iter' in ckpt_dict:
        return ckpt_dict['n_iter']
    if 'iteration' in ckpt_dict:
        return ckpt_dict['iteration']
    return 0


@AutoInjectConfigParams()
def load_ckpt(ckpt_name, models, optimizers=None, map_location="cuda", strict=False):
    try:
        if os.path.isdir(ckpt_name):
            ckpt_files = os.listdir(ckpt_name)
            assert ckpt_files is not None
            ckpt_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            ckpt_name = os.path.join(ckpt_name, ckpt_files[-1])

        ckpt_dict = torch.load(ckpt_name, map_location=map_location)
        for key, model in models:
            assert isinstance(model, nn.Module)
            model.load_state_dict(ckpt_dict[key], strict=strict)
        if optimizers is not None:
            for prefix, optimizer in optimizers:
                optimizer.load_state_dict(ckpt_dict[prefix])
    except:
        print(f"[!] LOAD FAIL")
        raise ValueError(f"model got Wrong checkpoint path: {ckpt_name}")

    print(f"[*] LOAD SUCCESS, model at path {ckpt_name} has loaded.")
    if 'n_iter' in ckpt_dict:
        return ckpt_dict['n_iter']
    if 'iteration' in ckpt_dict:
        return ckpt_dict['iteration']
    return 0


def default(vdict, key, default_value):
    if key in vdict:
        return vdict[key]
    return default_value


@StrictType
def load_model(model: nn.Module, model_config=None):
    default_ckpt_load_conf: GDict = GDict({
        'model_key': None,
        'name_mapping': None,
        'strict': False,
        'ignore_not_exist': False,
        'model': model,
        'ckpt_path': None
    })
    if model_config is None:
        model_config = GDict()
    default_ckpt_load_conf.update(model_config)
    model_config = default_ckpt_load_conf
    ckpt_path = model_config.ckpt_path
    if not os.path.isabs(ckpt_path):
        model_config.update({
            'ckpt_path': os.path.join(model_config.ckpt_folder, ckpt_path)
        })

    return load_checkpoint(model_config)


@StrictType
def set_requires_grad(net: nn.Module, requires_grad: bool):
    for param in net.parameters():
        param.requires_grad = requires_grad
