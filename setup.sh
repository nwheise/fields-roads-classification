#!/bin/bash

mkdir data
tar -xzf data.tar.gz -C data
rm data.tar.gz
conda env update --file env.yml