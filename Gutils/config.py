class GDict(dict):
    def __getattr__(self, item):
        return dict.__getitem__

    __setattr__ = dict.__setitem__
