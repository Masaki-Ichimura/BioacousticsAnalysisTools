import pathlib
from setuptools import setup, find_packages

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