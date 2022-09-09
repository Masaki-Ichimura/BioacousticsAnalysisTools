APP_NAME=BAGUI

git clone https://github.com/kivy/kivy-sdk-packager.git
cd kivy-sdk-packager/osx

curl -O -L https://github.com/kivy/kivy/releases/download/2.1.0/Kivy.dmg
hdiutil attach Kivy.dmg -mountroot .

cp -R Kivy/Kivy.app $APP_NAME.app
./fix-bundle-metadata.sh $APP_NAME.app -n MyApp -v "0.0.1" -a "Masaki-Ichimura" -o "ou.klab.myapp"

pushd $APP_NAME.app/Contents/Resources/venv/bin
source activate
popd

# install requirement packages
PYTHON_LIB_DIR=`python -c "import site; print(site.getsitepackages()[0])"`

python -m pip install ../../
git clone https://github.com/kivy-garden/garden.matplotlib.git $PYTHON_LIB_DIR/kivy/garden/matplotlib

# Reduce app size
./cleanup-app.sh $APP_NAME.app

# the link needs to be created relative to the yourapp path, so go to that directory
pushd $APP_NAME.app/Contents/Resources/
ln -s ./venv/bin/bagui yourapp
popd

./relocate.sh $APP_NAME.app
./create-osx-dmg.sh $APP_NAME.app $APP_NAME