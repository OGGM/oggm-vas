# flake8: noqa
from oggm import cfg

_doc = ('The VAS model output, with the specific mass balance, the terminus '
        'and maximum surface elevation and the response time scales on top '
        'of the geometry stored in `model_diagnostics`.')
cfg.add_to_basenames('vas_diagnostics', 'vas_diagnostics.nc', docstr=_doc)

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('oggm-vas')
except PackageNotFoundError:
    # package is not installed
    pass
finally:
    del version, PackageNotFoundError

from .core import *
