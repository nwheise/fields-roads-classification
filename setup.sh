mkdir data
tar -xvzf data.tar.gz -C data
rm data.tar.gz
conda env update --file env.yml
source activate bil