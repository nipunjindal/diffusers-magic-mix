#!/usr/bin/env bash

# Create virtual environment if it doesn't exist
if [ ! -d "magic-mix-env" ]; then
    python3 -m venv "magic-mix-env"
fi

# Activate virtual environment
source magic-mix-env/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install --install-hooks