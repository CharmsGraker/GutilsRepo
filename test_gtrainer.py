from Gutils.trainer import GTrainer

config_yml = 'config.yaml'

if __name__ == '__main__':
    # 会自动加载所有模型, 具体定义方式见config.yaml
    gtrainer = GTrainer(config_yml)
