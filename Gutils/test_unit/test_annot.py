from time import sleep

from omegaconf import OmegaConf

from Gutils.annot import AutoInjectConfigParams
from Gutils.objects import Functor


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


def test_on_interrupted():
    class F(Functor):
        def onInterrupted(self):
            print("onInterrupted")

        def f(self):
            while True:
                print("haha")
                sleep(0.1)

    F().run()


def test_case():
    @AutoInjectConfigParams()
    def ff(a, b=3, c=4, d=5):
        print(a, b, c, d)

    ff({
        'a': 123,
        'c': 23
    })


def test_unpack():
    @AutoInjectConfigParams()
    def zero_arg_f():
        print("zero_arg_f")

    @AutoInjectConfigParams()
    def single_arg_f(a: dict):
        print("single_arg_f: ", a)

    zero_arg_f()
    single_arg_f({2: 3})
    print("[PASS] test_single_arg_case")


if __name__ == '__main__':
    test_yml = '../../test_yml/test_1.yaml'

    cfg = OmegaConf.load(test_yml)
    test_unpack()
    test_case()
    test_case1(cfg)
    test_on_interrupted()