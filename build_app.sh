APP_NAME=BAGUI

if [ -e kivy-sdk-packager ]; then
    rm -rf kivy-sdk-packager/osx/$APP_NAME.{app,dmg}
else
    git clone https://github.com/kivy/kivy-sdk-packager.git
    curl -L https://github.com/kivy/kivy/releases/download/2.1.0/Kivy.dmg -o kivy-sdk-packager/osx/Kivy.dmg
fi

cd kivy-sdk-packager/osx

hdiutil attach Kivy.dmg -mountroot .

cp -R Kivy/Kivy.app $APP_NAME.app
diskutil unmount Kivy

./fix-bundle-metadata.sh $APP_NAME.app -n $APP_NAME -v "0.0.1" -a "Masaki-Ichimura" -o "ou.klab.myapp"

pushd $APP_NAME.app/Contents/Resources/venv/bin
source activate
popd

# install requirement packages
python -m pip install -U pip
python -m pip install -I ../../

# Reduce app size
./cleanup-app.sh $APP_NAME.app

# the link needs to be created relative to the yourapp path, so go to that directory
pushd $APP_NAME.app/Contents/Resources/
ln -s ./venv/bin/bagui yourapp
popd

./relocate.sh $APP_NAME.app

# enlarge size to avoid error (XX: 99=>192)
# create-osx-dmg.sh[line 73]: expr "$(cat "$work_dir/_size")" + XX > "$work_dir/_size"
sed -i '' '73s/99/192/g' create-osx-dmg.sh
./create-osx-dmg.sh $APP_NAME.app $APP_NAME