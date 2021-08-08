# import importlib
# import pkgutil
# __all__ = []
# for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
#     module = importlib.import_module('.'+module_name,package=__name__)
#     try: 
#         globals().update({k: getattr(module, k) for k in module.__all__})
#         __all__ += module.__all__
#     except AttributeError: continue
# # concatenate the __all__ from each of the submodules (expose to user)

__version__ = '1.0.1'
from .nn import *
from .groups import *
from .reps import *
