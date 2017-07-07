wordvec:
	echo "Training on wiki-en.txt with parameters: -min_count 5 -outputw vectorsW.txt -outputc vectorsC.txt -outputwc vectorsB.txt -cbow 0 -size 200 -window 10 -negative 5"
	time wordvec/word2vec -train wiki-en.txt -min_count 5 -outputw vectorsW.txt -outputc vectorsC.txt -outputwc vectorsB.txt -cbow 0 -size 200 -window 10 -negative 5
all:
	python main.py
