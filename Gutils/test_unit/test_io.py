from torch import nn

from Gutils.io import load_model


def test_load_model():
    m = nn.Module()
    load_model(m)


if __name__ == '__main__':
    test_load_model()
