from omegaconf import OmegaConf

from Gutils.annot import AutoInjectConfigParams


@AutoInjectConfigParams()
def ff(a, b=3):
    print(a, b)


@AutoInjectConfigParams()
def func2(t, bb, c=2800):
    print(t, bb, c)


def test_case1(cfg):
    d_conf = OmegaConf.create({
        't': 22
    })
    conf = OmegaConf.merge(d_conf, cfg)
    conf.update({'bb': 2})
    return func2(conf)


def test_case():
    ff({
        'a': 123,
        'c': 23
    })


if __name__ == '__main__':
    test_case()
    test_yml = '../../test_yml/test_1.yaml'

    cfg = OmegaConf.load(test_yml)
    test_case1(cfg)
