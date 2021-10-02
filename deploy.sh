#!bash

python3 setup.py bdist_wheel
twine upload dist/*

rm -rf build
rm -rf dist
