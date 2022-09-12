if [ ! -e venv ]; then
    sh install.sh
fi

pushd venv/bin
source activate
popd

bagui
