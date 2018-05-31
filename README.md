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

## Installation

A patched version of word2vec is included which allows accessing the context vectors
in Word2Vec.
