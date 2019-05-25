#!/bin/bash

mkdir data
tar -xzf data.tar.gz -C data
conda env update --file env.yml