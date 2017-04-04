echo "Example training for modified text2vec glove (with both word and context vectors dump)"
echo "Training on wiki-en.txt with parameters: input='wiki-en.txt' min_count=5 outputw='vectorsW.txt' outputc='vectorsC.txt' outputwc='vectorsB.txt' size=10 window=6"
time R CMD BATCH --no-save --no-restore '--args input='wiki-en.txt' min_count=5 outputw='vectorsW.txt' outputc='vectorsC.txt' outputwc='vectorsB.txt' size=10 window=6 ' ./trainGloveModle trainGloveModle.out&
