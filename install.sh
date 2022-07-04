PYTHON_VERSION='3.8.13'

# pyenv setup
# PLEASE run `pyenv install 3.8.13` in advance
pyenv local $PYTHON_VERSION

# venv
python -m venv env

# activate
. env/bin/activate

# upgrade pip
pip install --upgrade pip

# install python packages
PYTHON_LIB_DIR=`python -c "import site; print(site.getsitepackages()[0])"`
pip install -r requirements.txt
# `garden install matplotlib` が venv で使えないので，gitから取ってくる
git clone https://github.com/kivy-garden/garden.matplotlib.git $PYTHON_LIB_DIR/kivy/garden/matplotlib

# deactivate
deactivate
