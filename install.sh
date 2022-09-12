PYTHON_VERSION='3.9.13'

if [ ! -e venv ]; then
    # PLEASE run `pyenv install PYTHON_VERSION` in advance
    pyenv local $PYTHON_VERSION
    python -m venv venv
fi

pushd venv/bin
source activate
popd

pip install -U pip
pip install -I .
