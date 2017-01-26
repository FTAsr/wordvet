#comment
make
#if [ ! -e text8 ]; then
#  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
#  gzip -d text8.gz -f
#fi
time ./word2vec -train /Users/fa/workspace/repos/_codes/data/text8 -min_count 1 -outputw vectorsW.txt -outputc vectorsC.txt -outputwc vectorsB.txt -cbow 0 -size 14 -window 2 -negative 1 -hs 1 -sample 0 -threads 1 -binary 0 -iter 1
./distance vectors.bin
