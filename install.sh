PYTHON_VERSION='3.9.13'

if [ ! -e venv ]; then
    # PLEASE run `pyenv install 3.9.13` in advance
    pyenv local $PYTHON_VERSION
    python -m venv venv
fi

. venv/bin/activate

PYTHON_LIB_DIR=`python -c "import site; print(site.getsitepackages()[0])"`

pip install -U pip
pip install -I .
git clone https://github.com/kivy-garden/garden.matplotlib.git $PYTHON_LIB_DIR/kivy/garden/matplotlib

deactivate