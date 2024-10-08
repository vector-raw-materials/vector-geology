#!/bin/bash

source ~/.virtualenvs/gempy_2024.2/bin/activate

cd ../docs || exit
#make clean
make html
cd - || exit