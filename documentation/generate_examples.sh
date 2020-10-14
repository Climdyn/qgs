#!/usr/bin/env bash

cd source/files/examples || exit

for file in ./*.ipynb; do
    jupyter nbconvert --to rst "$file"
done
