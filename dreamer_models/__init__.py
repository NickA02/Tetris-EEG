from importlib import import_module
import inspect
import pkgutil

"""
Automatically re-export public classes and functions from submodules
in this package so they are available as: from experiment import SomeClass
"""


__all__ = []

# Iterate over modules in this package
for finder, module_name, ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
    module = import_module(module_name)

    # Export public classes and functions defined in the module
    for name, obj in inspect.getmembers(
        module, lambda o: inspect.isfunction(o) or inspect.isclass(o)
    ):
        if name.startswith("_"):
            continue
        # ensure the object is defined in that module (not imported)
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        globals()[name] = obj
        __all__.append(name)
