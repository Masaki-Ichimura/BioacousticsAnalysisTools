import pathlib
from os import system
from setuptools import setup, find_packages
from shutil import rmtree
from site import getsitepackages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name='batools',
    version='0.0.1',
    description='python based bio-acoustic analysis tools',
    author='Masaki-Ichimura',
    author_email='m.ichimura@gmail.com',
    url='https://github.com/Masaki-Ichimura/BioacousticsAnalysisTools',
    package_dir={'batools': 'batools'},
    package_data={
        'batools': list(map(str, (pathlib.Path.cwd()/'batools').glob(r'**/*.kv')))
    },
    packages=find_packages(
        include=['batools', 'batools.*'],
        exclude=['kivy-sdk-packager', 'kivy-sdk-packager.win']
    ),
    install_requires=install_requirements,
    entry_points={
        'console_scripts': ['bagui=batools.app.run:main',]
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ]
)

# This lines are instead of `garden install matplotlib` or `kivy garden install matplotlib`
# because there is no setup.py .
# It's recommended manual install for garden.matplotlib if you can because of setuptools architecture.
# If you choice manual install for garden.matplotlib, you nead modify part of build app script.
garden_matplotlib_url = 'https://github.com/kivy-garden/garden.matplotlib'
garden_matplotlib_dir = f'{getsitepackages()[0]}/kivy/garden/matplotlib'
rmtree(garden_matplotlib_dir, ignore_errors=True)
system(f'git clone {garden_matplotlib_url} {garden_matplotlib_dir}')
