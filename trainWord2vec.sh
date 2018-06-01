#!/bin/bash
set -eu
set -o pipefail
if [ "$#" -ne 3 ] ; then
  echo "word2vec_wrap <size> <window> <iters>"
  exit 0
fi
echo "Training for modified word2vec (with both word and context vectors dump)"
model_dir=models/word2vec_$1_$2
mkdir -p $model_dir
./word2vec/word2vec -train data/wiki.shuffled-norm1-phrase1 -min_count 1 -outputw $model_dir/vectorsW.txt -outputc $model_dir/vectorsC.txt -outputwc $model_dir/vectorsB.txt -cbow 0 -size $1 -window $2 -negative 5 -threads 20 -iter $3

