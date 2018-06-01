#!/bin/bash
set -eu
set -o pipefail
if [ "$#" -ne 3 ] ; then
  echo "word2vec_wrap <size> <window> <iters>"
  exit 0
fi
echo "Training for modified word2vec (with both word and context vectors dump)"
mkdir -p data/word2vec_$1_$2
./word2vec/word2vec -train wiki.shuffled-norm1-phrase1 -min_count 1 -outputw data/word2vec_$1_$2/vectorsW.txt -outputc data/word2vec_$1_$2/vectorsC.txt -outputwc data/word2vec_$1_$2/vectorsB.txt -cbow 0 -size $1 -window $2 -negative 5 -threads 20 -iter $3

