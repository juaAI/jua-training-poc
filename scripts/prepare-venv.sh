#!/usr/bin/env bash

set -euo pipefail

VENV_PATH=$1

poetry export --with=profiling --with=validation --format requirements.txt --without-hashes --output requirements-all.txt

# Remove editable requirements and local file:// references:
awk '!/\ @\ file:\/\//' requirements-all.txt | awk '!/^-e\ file:\/\/\/model\//' > requirements-external.txt

# Create virtual environment and install external requirements only:
python -m venv --clear $VENV_PATH
source $VENV_PATH/bin/activate
pip install -r ./requirements-external.txt
rm ./requirements-*.txt

# Deactivate virtual environment:
deactivate
