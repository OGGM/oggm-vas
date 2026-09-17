# flake8: noqa
from oggm import cfg

_doc = ('The VAS model output, with the specific mass balance, the terminus '
        'and maximum surface elevation and the response time scales on top '
        'of the geometry stored in `model_diagnostics`.')
cfg.add_to_basenames('vas_diagnostics', 'vas_diagnostics.nc', docstr=_doc)

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
