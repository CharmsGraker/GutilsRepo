import base64
import datetime
import inspect
import shutil
from typing import Union

import PIL

import io
import os.path
from importlib import import_module

import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig, ListConfig

from Gutils.annot import AutoInjectConfigParams, StrictType
from Gutils.config import GDict


class Inputs:
    pass

    def to_dict(self):
        D = {}
        for k, v in self.__dict__:
            D[k] = v
        return D


def get_all_methods(obj):
    m = {}
    for d in dir(obj):
        try:
            if inspect.ismethod(getattr(obj, d)):
                m[d] = getattr(obj, d)
        except:
            continue
    return m


class Outputs:
    def __init__(self):
        self.loss = torch.tensor(0., device='cuda', requires_grad=False)

    def _handle_default_key_(self, k, v):
        return v

    def add_item(self, k, v):
        setattr(self, k, v)

    def update(self, other):
        if isinstance(other, Outputs):
            other = other.to_dict()
        assert isinstance(other, dict)
        for k, v in other.items():
            if not hasattr(self, f'_handle_{k}_'):
                f = self._handle_default_key_
            else:
                f = getattr(self, f'_handle_{k}_')
            setattr(self, k, f(k, v))

    def update_loss_dict(self, loss_dict: dict, **kwargs):
        setattr(self, 'loss_dict', self._handle_loss_dict_(loss_dict, **kwargs))
        return self

    def __delattr__(self, item):
        if item == 'loss_dict':
            self.loss = 0.
        return super().__delattr__(item)

    def _handle_loss_dict_(self, other_loss_dict, weight=1., accumulate=False, detach=False):
        if not hasattr(self, 'loss_dict'):
            setattr(self, 'loss_dict', {})
        for k, v in other_loss_dict.items():
            if isinstance(v, torch.Tensor):
                if detach:
                    self.loss += v.item()
                else:
                    self.loss += v

        loss_dict = getattr(self, 'loss_dict')
        for k, v in other_loss_dict.items():
            if k not in loss_dict or not accumulate:
                loss_dict[k] = 0.
            loss_dict[k] += v * weight
        if accumulate:
            loss_dict['loss'] = self.loss
        return loss_dict

    def to_dict(self):
        D = {}
        for k, v in self.__dict__.items():
            D[k] = v
        return D


def find_class(location: str):
    ret = location.split('.')
    package = ".".join(ret[:-1])
    method = ret[-1]
    try:
        module = import_module(package)
        return getattr(module, method)
    except Exception as e:
        print('`find_class` raise exception: ', location)
        raise e


def base642img(imgbase64):
    img = io.BytesIO(base64.b64decode(imgbase64.encode('utf-8')))
    img = PIL.Image.open(img)
    return img


"""
    update dict with hierarchical structure
"""


def update_dot_dict(dot_dict, kv: dict, replace=False):
    for k, v in kv.items():
        cur_cfg = dot_dict
        split_keys = k.split('.')
        for i, dk in enumerate(split_keys):
            if i == len(split_keys) - 1:
                break
            if dk not in cur_cfg:
                cur_cfg[dk] = {}
            cur_cfg = cur_cfg[dk]
        # leaf node
        if split_keys[-1] in cur_cfg:
            if replace:
                del cur_cfg[split_keys[-1]]
                cur_cfg[split_keys[-1]] = v
        else:
            cur_cfg[split_keys[-1]] = v


def create_instance_from_config(config: Union[DictConfig, dict], *args, to_dict=True, **kwargs):
    """
        args-liked params passing is not preferred
    """
    assert isinstance(config, (DictConfig, dict))
    kind_key = 'kind' if 'kind' in config else 'target'

    T = config[kind_key]
    params = config['params'] if 'params' in config else {}
    if to_dict and isinstance(params, DictConfig):
        params = OmegaConf.to_object(params)
    update_dot_dict(params, kwargs, replace=True)
    obj = find_class(T)(*args, **params)
    obj.g_name = T.split('.')[-2:]
    return obj


def getObjectName(obj):
    import re
    ret = re.match("<class '*(.*?)'*>", str(obj.__class__))[1]
    return ret


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

    state_dict = remap_state_dict(state_dict, name_mapping)
    model.load_state_dict(state_dict, strict=strict)

    print(f"[*] LOAD SUCCESS, model at path {ckpt_name} has loaded.")
    if 'n_iter' in ckpt_dict:
        return ckpt_dict['n_iter']
    if 'iteration' in ckpt_dict:
        return ckpt_dict['iteration']
    return 0


def remap_state_dict(state_dict, name_mapping={}, use_dot_dict=True):
    if not isinstance(name_mapping, dict):
        name_mapping = {}
    import re

    if len(name_mapping) > 0:
        # rename parameters in ckpt, any else key not startswith prefix expected will be ignored
        new_ckpt_dict = {}
        for k, v in state_dict.items():
            need_replace = False
            for old_pre, new_pre in name_mapping.items():
                ret = re.match(old_pre, k)
                if ret is not None:
                    new_k = new_pre + k[len(old_pre):]
                    new_ckpt_dict[new_k] = v
                    need_replace = True
                    break
            # TODO: 可以只是更新state_dict，而不是只加载name_mapping中的state_dict
        state_dict = new_ckpt_dict
    return state_dict


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


def set_attr_if_absent(obj, attr, value, force=False):
    if force or not hasattr(obj, attr):
        setattr(obj, attr, value)


def set_if_absent(v, default_value=None):
    if v is None:
        return default_value
    return v


def default(obj, key, default_value):
    if isinstance(obj, (dict, DictConfig)):
        if key in obj:
            return obj[key]

    if key in obj.__dict__:
        return obj.__dict__[key]
    return default_value


@StrictType
def load_model(model: nn.Module, model_config=None):
    default_ckpt_load_conf: GDict = GDict({
        'model_key'       : None,
        'name_mapping'    : None,
        'strict'          : False,
        'ignore_not_exist': False,
        'model'           : model,
        'ckpt_path'       : None
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


def remove_old_file_if_necessary(d, old_than_given_seconds=3600 * 48  # unit: second
                                 ):
    if os.path.exists(d):
        ctime = get_file_datetime(d)
        now = datetime.datetime.now()
        stay_seconds = (now - ctime).seconds
        if stay_seconds > old_than_given_seconds:
            if os.path.isdir(d) and d != '/':
                shutil.rmtree(d)
            elif os.path.isfile(d):
                os.remove(d)
            print(f'remove {d} created at {ctime} ({stay_seconds} old than given criterion {old_than_given_seconds})')


def get_file_datetime(d, return_time='modified'):
    t = os.path.getmtime(d)
    creation_datetime = datetime.datetime.fromtimestamp(t)
    # formatted_creation_time = creation_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return creation_datetime


def select_best_ckpt(ckpt_list, monitor_meta):
    monitor = monitor_meta['monitor']
    mode = monitor_meta['mode']

    def filter_fn(x):
        import re
        try:
            ret = re.match(f'(.*?)-({monitor}=(\d+\.?(\d+)?))(-.*?)*(\.\w+)', x)
            metric = float(ret[3])
            return metric
        except BaseException as e:
            return 1e5 if mode == 'min' else -1e5

    best_path = sorted(ckpt_list, key=filter_fn, reverse=mode == 'max')[0]
    return best_path


if __name__ == '__main__':
    cl = ['../epoch=15-train_loss=1.1-ha=2.ckpt',
          '../epoch=15-train_loss=0.9-ha=2.ckpt',
          '../epoch=15-train_loss=0.019-ha=2.ckpt',
          'last.ckpt',
          'epoch=15-train_loss=110.123-hahha.ckpt']
    print(select_best_ckpt(cl, {
        'monitor': 'train_loss',
        'mode'   : 'min'
    }))
