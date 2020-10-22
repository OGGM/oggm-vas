# flake8: noqa
from oggm import cfg
_doc = ("A dict containing the glacier's t*, bias, mu*. Analogous "
        "to 'local_mustar.json', but for the volume/area scaling model.")
cfg.add_to_basenames('vascaling_mustar', 'vascaling_mustar.json', docstr=_doc)

try:
    from .version import version as __version__
    from .version import isreleased as __isreleased__
except ImportError:
    raise ImportError('oggm-vas is not properly installed. If you are '
                      'running from the source directory, please instead '
                      'create a new virtual environment (using conda or '
                      'virtualenv) and  then install it in-place by running: '
                      'pip install -e .')
from .core import *
