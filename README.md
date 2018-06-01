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
```

## Running experiments

Once all models are trained and placed in `data/`, run the experiments with

````bash
python3 main.py
````

This will emit scores for Similarity and Relatedness for each family of models. For
example this is how output might look for a word2vec model with vector size 100 and
window size of 5

````
Loading model from: data/word2vec_100_5
Loading W model...
Loading C model...
**** SIMILARITY ****
FA: similarity calculation...
0 pairs processed!
['old', 'new', '1.58']
word0 = old
word1 = new
(984, 7)
0.065961898938654
             Wordpair  GoldSimilarity        WW        CC        BB        WC        CW  predicted
0             old-new            1.58  0.574986  0.359083  0.338911 -0.075662 -0.107971   3.217153
1   smart-intelligent            9.20  0.877416  0.796356  0.307454 -0.718494 -0.798851   4.814335
2      hard-difficult            8.77  0.854869  0.842902  0.768026 -0.061308 -0.041688   5.099599
3      happy-cheerful            9.55  0.844289  0.717220  0.302065 -0.784213 -0.598359   4.521168
4           hard-easy            0.95  0.892240  0.797713  0.765535 -0.099818 -0.054987   4.973763
5          fast-rapid            8.75  0.824911  0.712174  0.529688 -0.316606 -0.327214   4.584165
6          happy-glad            9.17  0.894823  0.828813  0.704390 -0.662424 -0.552113   5.190273
7          short-long            1.23  0.835283  0.773027  0.748878  0.047126  0.010861   4.842519
8         stupid-dumb            9.58  0.967084  0.956882  0.652549 -0.864821 -0.898794   5.653663
9       weird-strange            8.93  0.881273  0.812451 -0.098299 -0.834247 -0.890774   4.553763
10        wide-narrow            1.03  0.930155  0.838667  0.845991 -0.127527 -0.125285   5.193918
11          bad-awful            8.42  0.799531  0.694747  0.280337 -0.721824 -0.623955   4.424680
12     easy-difficult            0.58  0.891146  0.817458  0.829969 -0.019838  0.013872   5.064387
13       bad-terrible            7.78  0.843449  0.825382  0.670940 -0.532189 -0.487958   5.113020
14        hard-simple            1.38  0.817010  0.734233  0.671131 -0.072461 -0.070585   4.685422
15         smart-dumb            0.55  0.960945  0.933039  0.370990 -0.898263 -0.904275   5.345687
16       insane-crazy            9.57  0.941637  0.908984  0.557712 -0.802614 -0.836516   5.400948
17          happy-mad            0.95  0.869143  0.843449  0.677158 -0.586675 -0.601239   5.210555
18         large-huge            9.47  0.850300  0.669710  0.680622 -0.209200 -0.032159   4.507737
19         hard-tough            8.05  0.848237  0.591574  0.637592 -0.617705 -0.335191   4.343801

Correlation bw AllReg and gold:SpearmanrResult(correlation=0.22754160153145972, pvalue=5.063281745085291e-13)
 Correlation bw WW and gold:0.20080835293770594
 Correlation bw CC and gold:0.21955558182627896
 Correlation bw AA and gold:0.1731680215990427
 Correlation bw WC and gold:0.01729646059265929
 Correlation bw CW and gold:0.012455066579276301
**** RELATEDNESS McRae ****
FA: similarity calculation...
0 pairs processed!
['AIRPLANE', 'pilot', '88', '']
word0 = airplane
word1 = pilot
(1032, 7)
0.010598076564171222
              Wordpair  GoldSimilarity        WW        CC        BB        WC        CW  predicted
0       airplane-pilot              88  0.909827  0.860755  0.827478 -0.469825 -0.547111  43.446211
1      airplane-flight              77  0.849144  0.792775  0.807901 -0.379538 -0.427566  42.286663
2         airplane-sky              61  0.781424  0.604982  0.193000 -0.587097 -0.612948  40.846469
3       airplane-cloud              41  0.886239  0.803825  0.153972 -0.806438 -0.791131  40.131872
4     airplane-airport              41  0.729982  0.628604  0.338502 -0.496211 -0.496086  38.248354
5      airplane-travel              32  0.797149  0.639978  0.467962 -0.460716 -0.515977  42.114116
6       airplane-wings              30  0.887456  0.736147  0.571294 -0.528648 -0.595704  44.224893
7   airplane-passenger              24  0.898700  0.863854  0.599387 -0.695132 -0.620992  39.336796
8        airplane-food              18  0.685123  0.472939  0.124395 -0.474463 -0.548418  39.998689
9       airplane-seats              17  0.729266  0.593020  0.336157 -0.445572 -0.509053  39.994659
10     airplane-runway              15  0.963775  0.901837  0.612025 -0.836308 -0.768690  40.396421
11    airplane-luggage              14  0.948267  0.827351 -0.064457 -0.928912 -0.863643  41.647330
12          apple-tree              73  0.522789  0.347169 -0.065603 -0.446127 -0.548840  34.770877
13           apple-red              40  0.418642  0.313789  0.003543 -0.339672 -0.385244  30.484212
14           apple-pie              37  0.664426  0.613749 -0.229751 -0.751553 -0.695440  33.104340
15        apple-orange              32  0.657600  0.561054  0.174200 -0.488055 -0.441784  35.736803
16         apple-fruit              32  0.558487  0.373025 -0.062734 -0.500947 -0.545994  34.996904
17          apple-worm              29  0.736840  0.698428  0.186474 -0.667999 -0.636568  35.533431
18        apple-banana              24  0.639838  0.598387 -0.089380 -0.651095 -0.686858  33.741263
19          apple-core              23  0.694425  0.537349  0.194175 -0.442059 -0.478201  39.289281

Correlation bw AllReg and gold:SpearmanrResult(correlation=0.08358718899531849, pvalue=0.007216893658738309)
 Correlation bw WW and gold:0.03534104270367062
 Correlation bw CC and gold:0.038971640460957774
 Correlation bw AA and gold:0.0818613503609274
 Correlation bw WC and gold:0.06694020809976141
 Correlation bw CW and gold:0.05537049463389212
````
