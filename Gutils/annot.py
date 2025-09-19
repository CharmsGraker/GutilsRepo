import copy
import functools
import inspect

from omegaconf import DictConfig

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


"""
自动将字典式的输入unpack 送进被调用的函数，如果函数没有形参或形参数==1，则不会把dict或DictConfig拆包
"""


def AutoInjectConfigParams(suppressRedundantParams=True):
    verbose = True

    def decorate(f):
        @functools.wraps(f)
        def deco(*args, **kwargs):
            signature = inspect.signature(f)
            if len(signature.parameters) > 1:
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


def suppress_kwargs_redundant():
    def decorate(f):
        @functools.wraps(f)
        def deco(*args, **kwargs):
            signature = inspect.signature(f)

            if kwargs is not None:
                inject_var_args = ()
                arg_ptr = 0
                inject_kwargs = kwargs
                sig_params = signature.parameters

                # remove all unwanted params
                _tmp_kwargs = copy.deepcopy(inject_kwargs)
                for k, v in inject_kwargs.items():
                    if k not in sig_params:
                        del _tmp_kwargs[k]
                inject_kwargs = _tmp_kwargs
                inject_var_kwargs = {}
                for name, p in sig_params.items():
                    # handle arg-like arguments
                    if p.kind == p.VAR_KEYWORD:
                        inject_var_kwargs[name] = kwargs[name]
                    elif p.kind == p.VAR_POSITIONAL:
                        if len(args) > arg_ptr:
                            inject_var_args = args[arg_ptr:]
                    else:  # positional_or_keyword
                        # args
                        if p.default is inspect.Parameter.empty:  # args, e.g., `self`
                            if name not in inject_kwargs:  # reformulate it as kwargs
                                inject_kwargs[name] = args[arg_ptr]
                            arg_ptr += 1
                        else:  # kwargs
                            # no need to handle kwargs-like arguments
                            # refill the parameter with `name` and  without passing argument explict with default value
                            # set default value for this dict-like argument
                            # if name not in inject_kwargs:
                            #     inject_kwargs[name] = p.default
                            pass
                return functools.partial(f, *inject_var_args, **inject_kwargs, **inject_var_kwargs)()
            return f(*args, **kwargs)

        return deco

    return decorate


if __name__ == '__main__':
    pass
