from collections import defaultdict

__all__ = ['list_models']

_model_registry = [] # mapping of model names to entrypoint fns


def register_model():
    def inner_decorator(fn):
        # add entries to registry dict/sets
        model_name = fn.__name__
        _model_registry.append(model_name)
        return fn
    return inner_decorator


def list_models():
    """ Return list of available model names
    """
    return _model_registry