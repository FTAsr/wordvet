make

sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < wiki.txt | tr -c "A-Za-z'_ \n" " " > wiki.shuffled-norm0
time ./word2phrase -train wiki.shuffled-norm0 -output wiki.shuffled-norm0-phrase0 -threshold 200 -debug 2
time ./word2phrase -train wiki.shuffled-norm0-phrase0 -output wiki.shuffled-norm0-phrase1 -threshold 100 -debug 2
tr A-Z a-z < wiki.shuffled-norm0-phrase1 > wiki.shuffled-norm1-phrase1
time ./word2vec -train wiki.shuffled-norm1-phrase1 -min_count 1 -outputw vectorsW.txt -outputc vectorsC.txt -outputwc vectorsB.txt -cbow 0 -size 100 -window 6 -negative 1 -hs 1 -sample 0 -threads 1 -binary 0 -iter 1
./distance vectors.bin
