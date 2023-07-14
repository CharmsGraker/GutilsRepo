import math

from omegaconf import OmegaConf, DictConfig
import hydra



def generate_template_config(config_path):
    pass


OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))

config_yml = 'config.yaml'
oconf = OmegaConf.load(config_yml)

print(oconf)
# 注意，直接打印conf不会渲染值
print(oconf['models']['params']['ss'])
print(oconf.models.params.ss)

# config_path 配置绝对路径的parent dir
@hydra.main(version_base=None, config_path='..', config_name='config')
def injected_app(cfg: DictConfig):
    print(cfg)
    print(cfg.models.params.ss)


if __name__ == '__main__':
    injected_app()
