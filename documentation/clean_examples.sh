#!/usr/bin/env bash

cd source/files/examples

for file in ./*.ipynb; do
    rm -f ${file%.*}".rst"
done

