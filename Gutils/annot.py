import copy
import functools
import inspect

from omegaconf import DictConfig, OmegaConf


"""
StrictType
会对函数入参指定型别的参数进行检查，若实参的型别与形参型别不一致，则不通过校验
"""
def StrictType(f):
    @functools.wraps(f)
    def deco(*args, **kwargs):
        signature = inspect.signature(f)
        sparams = signature.parameters
        total = len(sparams)
        param_keys = list(sparams.keys())

        i = 0
        for a in args:
            k = param_keys[i]

            if sparams[k].annotation != sparams[k].empty and type(a) != sparams[k].annotation:
                raise Exception(f'params at args {i} type mismatched!')
            i += 1
        for k, v in kwargs.items():
            if sparams[k].annotation != sparams[k].empty and type(v) != sparams[k].annotation:
                raise Exception(f'params at kwargs {i} type mismatched!')
        return f(*args, **kwargs)

    return deco


def AutoInjectConfigParams(suppressRedundantParams=True):
    verbose = True

    def decorate(f):
        @functools.wraps(f)
        def deco(*args, **kwargs):
            dictParams = None
            if len(args) == 0:
                dictParams = kwargs
            else:
                if isinstance(*args, DictConfig):
                    dictParams = args[0]
                elif isinstance(*args, dict):
                    dictParams = args[0]

            if dictParams is not None:
                inject_kwargs = dictParams
                signature = inspect.signature(f)
                sparams = signature.parameters
                for k, v in sparams.items():
                    if k not in inject_kwargs:
                        if v.default is inspect.Parameter.empty:
                            raise Exception(
                                f"error occur when execute {inspect.getfile(f)}\n{inspect.getsource(f)}\nmissing required params '{k}'")
                        inject_kwargs[k] = v.default
                tmp_kwargs = copy.deepcopy(inject_kwargs)
                if suppressRedundantParams:
                    for k, v in inject_kwargs.items():
                        if k not in sparams:
                            del tmp_kwargs[k]
                    inject_kwargs = tmp_kwargs
                return functools.partial(f, **inject_kwargs)()
            return f(*args, **kwargs)

        return deco

    return decorate
