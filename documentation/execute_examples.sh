#!/usr/bin/env bash

cd source/files/examples || exit

for file in ./*.ipynb; do
    jupyter nbconvert --execute --inplace "$file"
done

