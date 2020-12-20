#!/usr/bin/env bash

cd source/files/examples || exit

for file in ./*.ipynb; do
    rm -f ${file%.*}".rst"
done

