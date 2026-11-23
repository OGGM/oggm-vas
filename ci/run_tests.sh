#!/bin/bash
# Install oggm-vas together with the current OGGM master and run the tests.
set -e
set -x

chown -R "$(id -u):$(id -g)" "$HOME"

export MPLBACKEND=agg

# `python -m` throughout: the console scripts are not on PATH in all images
PYTHON=${PYTHON:-python3}

# oggm-vas tracks OGGM master, not the latest release
$PYTHON -m pip install --upgrade git+https://github.com/fmaussion/salem.git
$PYTHON -m pip install --upgrade git+https://github.com/OGGM/oggm.git
$PYTHON -m pip install -e ".[tests]"

$PYTHON -m pytest --verbose --durations=10 oggm_vas
