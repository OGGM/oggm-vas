#!/bin/bash
# Install oggm-vas together with the current OGGM master and run the tests.
set -e
set -x

chown -R "$(id -u):$(id -g)" "$HOME"

export MPLBACKEND=agg

PIP=${PIP:-pip3}

# oggm-vas tracks OGGM master, not the latest release
$PIP install --upgrade git+https://github.com/fmaussion/salem.git
$PIP install --upgrade git+https://github.com/OGGM/oggm.git
$PIP install -e ".[tests]"

pytest --verbose --durations=10 oggm_vas
