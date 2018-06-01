# Querying Word Embeddings for Similarity and Relatedness

This repository contains the code for our NAACL 2018 paper:

*[Querying Word Embeddings for Similarity and Relatedness](http://aclweb.org/anthology/N18-1062)*.

If you use this software please cite:

````
@inproceedings{asr2018embeddings,
  author =      {Fatemeh Torabi Asr and Robert Zinkov and Michael N. Jones},
  title =       {Querying Word Embeddings for Similarity and Relatedness},
  booktitle =   {Proceedings of the 2018 Conference of the North
                 American Chapter of the Association for Computational
                 Linguistics: Human Language Technologies (NAACL-HLT)},
  year =        {2018},
  url =         {http://aclweb.org/anthology/N18-1062},
  publisher =   {Association for Computational Linguistics},
  pages =       {675--684}
}
````

## Dependencies
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [PyYaml](https://github.com/yaml/pyyaml)
- [pandas](https://github.com/pandas-dev/pandas)
- [text2vec](https://github.com/dselivanov/text2vec)

## Installation

A patched version of word2vec is included which allows accessing the context vectors
in Word2Vec. Install it locally

````bash
cd word2vec
make
cd ..
````

## Downloading training data

We use the wikipedia dumps from the [Polyglot project](https://sites.google.com/site/rmyeid/projects/polyglot#TOC-Download-Wikipedia-Text-Dumps)

Once they have been downloaded, preprocess it so there is one sentence per-line

````bash
tar --lzma -xvf en_wiki_text.tar.lzma
mv en/full.txt wiki.txt
sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < wiki.txt | tr -c "A-Za-z'_ \n" " " > wiki.shuffled-norm0
time ./word2vec/word2phrase -train wiki.shuffled-norm0 -output wiki.shuffled-norm0-phrase0 -threshold 200 -debug 2
time ./word2vec/word2phrase -train wiki.shuffled-norm0-phrase0 -output wiki.shuffled-norm0-phrase1 -threshold 100 -debug 2
tr A-Z a-z < wiki.shuffled-norm0-phrase1 > wiki.shuffled-norm1-phrase1
mv wiki.shuffled-norm1-phrase1 data/
````

## Building models

Word2vec models can be built with the `trainWord2Vec.sh` script

````bash
# Train word2vec model with vector size 100, window size 5 for 2 iterations
bash trainWord2Vec.sh 100 5 2
````

GloVe models can be built with the `trainGloveModel.r` script

````bash
# Train GloVe model with vector size 100, skipgram window size 5 for iterations
Rscript trainGloveModel.r 100 5 2
````

## Running experiments

Once all models are trained and placed in `data/`, run the experiments with

````bash
python3 main.py
````

This will emit scores for Similarity and Relatedness for each family of models. For
example this is how output might look for a word2vec model with vector size 300 and
window size of 3

````
Loading model from: /data/model_300_3
Loading W model...
Loading C model...
**** SIMILARITY ****
FA: similarity calculation...
0 pairs processed!
['old', 'new', '1.58']
word0 = old
word1 = new
(999, 7)
0.211520439143
             Wordpair  GoldSimilarity        WW        CC        BB        WC  \
0             old-new            1.58  0.418562  0.353129  0.360353  0.029681   
1   smart-intelligent            9.20  0.645673  0.658443  0.636772  0.245393   
2      hard-difficult            8.77  0.665176  0.661634  0.613458  0.090829   
3      happy-cheerful            9.55  0.616203  0.664373  0.606519  0.147825   
4           hard-easy            0.95  0.580208  0.600023  0.519916  0.061451   
5          fast-rapid            8.75  0.580806  0.572373  0.547164  0.186563   
6          happy-glad            9.17  0.654879  0.697013  0.617484  0.053706   
7          short-long            1.23  0.646191  0.607111  0.623390  0.116351   
8         stupid-dumb            9.58  0.715795  0.759943  0.721452  0.241709   
9       weird-strange            8.93  0.772030  0.787016  0.749360  0.230786   
10        wide-narrow            1.03  0.636410  0.641876  0.621156  0.182498   
11          bad-awful            8.42  0.632052  0.676788  0.603146  0.079041   
12     easy-difficult            0.58  0.650389  0.678808  0.594419  0.068222   
13       bad-terrible            7.78  0.651089  0.672962  0.587841  0.070799   
14        hard-simple            1.38  0.341476  0.384788  0.267727 -0.016254   
15         smart-dumb            0.55  0.515982  0.564391  0.472334  0.096045   
16       insane-crazy            9.57  0.551453  0.565938  0.495667  0.095963   
17          happy-mad            0.95  0.446828  0.514721  0.381354 -0.004133   
18         large-huge            9.47  0.774587  0.776006  0.711643 -0.007756   
19         hard-tough            8.05  0.640310  0.658540  0.613479  0.147848   

          CW  predicted  
0   0.009628   5.304092  
1   0.219546   6.245192  
2   0.093109   6.401615  
3   0.163010   5.596971  
4   0.063084   5.548970  
5   0.201193   5.835302  
6   0.101871   5.799755  
7   0.108661   6.709913  
8   0.239773   6.411573  
9   0.209378   7.085886  
10  0.180708   6.186711  
11  0.155024   5.552260  
12  0.049463   6.029783  
13  0.111595   5.923467  
14 -0.016587   3.595965  
15  0.121621   4.776221  
16  0.099893   5.392644  
17  0.022385   4.054693  
18  0.039315   7.013737  
19  0.184552   5.966043  

Correlation bw AllReg and gold:SpearmanrResult(correlation=0.45947031512025005, pvalue=2.5147923077595237e-53)
 Correlation bw WW and gold:0.437658523445
 Correlation bw CC and gold:0.404676160374
 Correlation bw AA and gold:0.423223540301
 Correlation bw WC and gold:0.346526093211
 Correlation bw CW and gold:0.324066293496
**** RELATEDNESS McRae ****
FA: similarity calculation...
0 pairs processed!
['AIRPLANE', 'pilot', '88', '']
word0 = airplane
word1 = pilot
(1106, 7)
0.0761163954799
              Wordpair  GoldSimilarity        WW        CC        BB  \
0       airplane-pilot              88  0.551976  0.600694  0.544371   
1      airplane-flight              77  0.648705  0.678587  0.649902   
2         airplane-sky              61  0.322724  0.386831  0.257835   
3       airplane-cloud              41  0.285508  0.379588  0.196987   
4     airplane-airport              41  0.482195  0.454481  0.400134   
5      airplane-travel              32  0.291911  0.333209  0.269403   
6       airplane-wings              30  0.421554  0.425669  0.375780   
7   airplane-passenger              24  0.481048  0.515415  0.475352   
8        airplane-food              18  0.194619  0.259811  0.056630   
9       airplane-seats              17  0.236049  0.268263  0.177089   
10     airplane-runway              15  0.455255  0.487825  0.406472   
11    airplane-luggage              14  0.440677  0.501814  0.402660   
12          apple-tree              73  0.389762  0.402236  0.321588   
13           apple-red              40  0.336349  0.296563  0.262519   
14           apple-pie              37  0.337035  0.416692  0.298411   
15        apple-orange              32  0.442255  0.428381  0.375702   
16         apple-fruit              32  0.457979  0.459386  0.398294   
17          apple-worm              29  0.435112  0.486264  0.364163   
18        apple-banana              24  0.495931  0.496723  0.438851   
19          apple-core              23  0.245878  0.258216  0.164963   

          WC        CW  predicted  
0   0.185927  0.135477  59.683689  
1   0.218118  0.140787  70.663772  
2  -0.000399 -0.018288  35.487846  
3  -0.071328 -0.055097  23.579947  
4   0.051316  0.030929  51.116467  
5   0.040935  0.019152  39.725338  
6   0.098311  0.064568  50.680814  
7   0.137250  0.133410  50.005181  
8  -0.139273 -0.165157  23.107587  
9  -0.019191 -0.022723  31.904602  
10  0.067119  0.050088  46.674611  
11  0.066390  0.070446  41.345691  
12  0.072957  0.047687  45.545807  
13  0.072238  0.037722  48.253110  
14  0.041319  0.053099  33.300960  
15  0.108392  0.097271  49.194714  
16  0.096017  0.088466  48.226502  
17  0.043698  0.066364  37.048830  
18  0.097149  0.143880  42.780899  
19 -0.012895 -0.039956  36.212444  

Correlation bw AllReg and gold:SpearmanrResult(correlation=0.24471298396768196, pvalue=1.5197910479313451e-16)
 Correlation bw WW and gold:0.192516384248
 Correlation bw CC and gold:0.180317055598
 Correlation bw AA and gold:0.207177011707
 Correlation bw WC and gold:0.238727141468
 Correlation bw CW and gold:0.195237066341
````
