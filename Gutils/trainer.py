import os

import hydra
from omegaconf import DictConfig, OmegaConf

from Gutils.config import GDict
from Gutils.io import create_instance, set_requires_grad, load_model


class GTrainer:
    def __init__(self, main_config_path):
        self.device = 'cuda'
        assert main_config_path != None
        if not os.path.isabs(main_config_path):
            pwd = os.path.abspath(os.getcwd())
            main_config_path = os.path.join(pwd, main_config_path)
            if not os.path.exists(main_config_path):
                raise FileNotFoundError(f"can not found config in {main_config_path}")
        print("trying read config in " + main_config_path)

        @hydra.main(version_base=None, config_path=os.path.dirname(main_config_path),
                    config_name=os.path.basename(main_config_path))
        def load_config(cfg: DictConfig):
            setattr(self, 'config', cfg)

        load_config()
        print(self.config)
        self.drop_meta = True
        self.ckpt_folder = None
        self.configure_models(self.get_config())

    def get_config(self):
        return self.config

    def configure_models(self, cfg):
        # 由于OmegaConf不支持非原始型别，这里为每个模型再增加一个层级，meta用于存储omegaconf对象, others存储其他非原型对象
        model_dict = GDict()
        models_cfg: DictConfig = cfg.models
        template_model_config = OmegaConf.create({
            'params': {},
            'ckpt_iter': 0,
            'requires_grad': True,

        })
        for p, model_config in models_cfg.items():
            tmp = {}
            Type = model_config['kind'] if 'kind' in model_config else model_config['target']
            # due to pl eager behavior, we can not inject device here, will create these model later
            tmp['model_type'] = Type
            for attr_name, val in model_config.items():
                tmp[attr_name] = val
            model_dict[p] = GDict
            model_dict[p].meta = OmegaConf.merge(template_model_config, OmegaConf.create(tmp))
            model_dict[p].others = None

        # create all model instances
        print()
        self.model_dict = model_dict
        for m_name, d_config in self.model_dict.items():
            omega_meta = d_config.meta
            Type = omega_meta.model_type
            instance = create_instance(Type, omega_meta.params)
            model_dict[m_name].state_dict = instance
            set_requires_grad(instance, omega_meta.requires_grad)
            model_dict[m_name].ckpt_iter = load_model(model_dict[m_name].others.state_dict, self.ckpt_folder,
                                                      omega_meta.checkpoint)

            model_dict[m_name].others.state_dict = instance.to(self.device)
            if self.drop_meta:
                setattr(self, m_name, d_config.others.state_dict)
                for attr_name, val in d_config.items():
                    if attr_name not in ['model_type', 'state_dict', 'params', 'kind', 'target']:
                        setattr(getattr(self, m_name), attr_name, val)
            print(f"{m_name}({Type}) has been created...")
