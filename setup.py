import pathlib
from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

exclude_packages = ['kivy-sdk-packager', 'kivy-sdk-packager.win']
packages = find_packages(include=['batools', 'batools.*'], exclude=exclude_packages)

setup(
    name="batools",
    version="0.0.1",
    description="BioacousticAnalysisTools",
    author="Masaki-Ichimura",
    packages=packages,
    package_dir={'batools': 'batools'},
    package_data={
        'batools': list(map(str, (pathlib.Path.cwd()/'batools').glob(r'**/*.kv')))
    },
    install_requires=install_requirements,
    entry_points={
        'console_scripts': ["myapp=batools.app.run:main",]
    },
    classifiers=['Programming Language :: Python :: 3.9']
)