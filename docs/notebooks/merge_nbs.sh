#!/bin/bash
#set -x
dir_path=$(dirname $(realpath $0))
PREAMBLE=$dir_path/_colab_preamble.ipynb
NOTEBOOKS=$dir_path/[^_]*.ipynb
mkdir  $dir_path/colabs
cp -r $dir_path/imgs  $dir_path/colabs/imgs
for nb in $NOTEBOOKS
do
    nbmerge $PREAMBLE $nb -o $dir_path/colabs/$(basename $nb)
done
nbmerge $PREAMBLE $NOTEBOOKS -o $dir_path/colabs/all.ipynb
# nbmerge -i notebooks/[^_]*.ipynb -o notebooks/_colab_all.ipynb