PYTHON_VERSION='3.9.13'

# pyenv setup
# PLEASE run `pyenv install 3.9.13` in advance
pyenv local $PYTHON_VERSION

# venv
python -m venv env

# activate
. env/bin/activate

# install python packages
PYTHON_LIB_DIR=`python -c "import site; print(site.getsitepackages()[0])"`
pip install --upgrade pip
pip install -r requirements.txt
# `garden install matplotlib` が venv で使えないので，gitから取ってくる
git clone https://github.com/kivy-garden/garden.matplotlib.git $PYTHON_LIB_DIR/kivy/garden/matplotlib

# deactivate
deactivate
