#!/usr/bin/env python

from __future__ import print_function
from gensim import matutils
from gensim.models import KeyedVectors
import numpy as np

import os
import sys

from scipy import spatial
from sklearn.preprocessing import normalize

import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy
from datetime import datetime


def similarities(wordpairFile, modelWord, modelContext,  outputFile):
    outf = open(outputFile, 'w')
    mw = modelWord
    mc = modelContext
    print("FA: similarity calculation...")
    outf.write("Wordpair"
               + "," + "GoldSimilarity"
               + "," + "GoldSimilarityBackward"
               + "," + "WW"
               + "," + "CC"
               + "," + "BB"
               + "," + "WC"
               + "," + "CW"
               + "\n")

    #wordpairs = [line.rstrip('\n') for line in open(wordpairFile)]
    wordpairs = open(wordpairFile).read().splitlines()
    for index, pair in enumerate(wordpairs):
        words = pair.split(',')
        if len(words) < 3:
            continue
        word0 = words[0].lower().replace(" ", "")
        word1 = words[1].lower().replace(" ", "")
        word2 = words[2].lower().replace(" ", "")
        if word2 == "no":
            continue
        if index % 10000 == 0:
            print(str(index) + " pairs processed!")
            print(words)
            print("word0 = " + word0)
            print("word1 = " + word1)

        if word0 in mw.vocab and word1 in mw.vocab:
            w0 = mw[word0]
            w1 = mw[word1]
            c0 = mc[word0]
            c1 = mc[word1]
            w0 = normalize(w0[:, np.newaxis], axis=0).ravel()
            w1 = normalize(w1[:, np.newaxis], axis=0).ravel()
            c0 = normalize(c0[:, np.newaxis], axis=0).ravel()
            c1 = normalize(c1[:, np.newaxis], axis=0).ravel()

            b0 = np.add(w0, c0)
            b1 = np.add(w1, c1)

            b0 = normalize(b0[:, np.newaxis], axis=0).ravel()
            b1 = normalize(b1[:, np.newaxis], axis=0).ravel()

            wwScore = 1 - spatial.distance.cosine(w0, w1)
            ccScore = 1 - spatial.distance.cosine(c0, c1)
            wcScore = 1 - spatial.distance.cosine(w0, c1)
            cwScore = 1 - spatial.distance.cosine(w1, c0)
            bbScore = 1 - spatial.distance.cosine(b0, b1)

            #  word pair and the gold similarity scores
            #  (specialized to nelson data)
            outf.write(word0 + "-" + word1 + "," + word2
                       + "," + str(wwScore)
                       + "," + str(ccScore)
                       + "," + str(bbScore)
                       + "," + str(wcScore)
                       + "," + str(cwScore)
                       + "\n")
        else:
            print("One of these words missing in our model:",
                  word0,
                  word1)
    outf.close()
    return similarities


def rank(m, targetWord, cueVector, restrict_vocab=None):
    """
    Find the rank of targetWord in the vocabulary with
    regards to its similarity to the cue vector
    """
    m.init_sims()
    mean = cueVector
    if restrict_vocab is None:
        limited = m.syn0norm
    else:
        limited = m.syn0norm[:restrict_vocab]
    dists = np.dot(limited, mean)
    best = matutils.argsort(dists, reverse=True)
    result = [(m.index2word[sim], float(dists[sim])) for sim in best]
    #print result[0]
    return [x for x, y in enumerate(result) if y[0] == targetWord]


def similaritiesAndRanks(wordpairFile,
                         modelWord,
                         modelContext,
                         outputFile,
                         n=50):
    mw = modelWord
    mc = modelContext
    outf = open(outputFile, 'w')
    print("FA: in ranksInNmostSimilar...")
    outf.write("Wordpair" + ","
               + "GoldSimilarity"
               + "," + "WW"
               + "," + "CC"
               + "," + "AA"
               + "," + "RW0W1"
               + "," + "RW1W0"
               + "," + "RC0C1"
               + "," + "RC1C0"
               + "\n")

    print("FA: dictionary of word:NmostSimilar is being made")
    print("Current time: " + str(datetime.now().time()))
    wordpairs = open(wordpairFile).read().splitlines()
    dictionaryC = dict()
    dictionaryW = dict()
    for index, pair in enumerate(wordpairs):
        words = pair.split(',')
        word0 = words[0].lower()
        word1 = words[1].lower()
        print(words)
        if word0 not in dictionaryW and word0 in mw.vocab:
            dictionaryW[word0] = mw.most_similar(word0, topn=n)
        if word1 not in dictionaryW and word1 in mw.vocab:
            dictionaryW[word1] = mw.most_similar(word1, topn=n)
        if word0 not in dictionaryC and word0 in mc.vocab:
            dictionaryC[word0] = mc.most_similar(word0, topn=n)
        if word1 not in dictionaryC and word1 in mc.vocab:
            dictionaryC[word1] = mc.most_similar(word1, topn=n)
    print("Now we have the",
          str(n),
          "most similar words to each word",
          "of the vocabulary in the test file!")
    print("Current time:", str(datetime.now().time()))

    for index, pair in enumerate(wordpairs):
        print("Writing wordpairs to the file...")
        words = pair.split(',')
        word0 = words[0].lower()
        word1 = words[1].lower()
        if word0 in mw.vocab and word1 in mw.vocab:
            w0 = mw[word0]
            w1 = mw[word1]
            c0 = mc[word0]
            c1 = mc[word1]
            w0 = normalize(w0[:, np.newaxis], axis=0).ravel()
            w1 = normalize(w1[:, np.newaxis], axis=0).ravel()
            c0 = normalize(c0[:, np.newaxis], axis=0).ravel()
            c1 = normalize(c1[:, np.newaxis], axis=0).ravel()
            b0 = np.add(w0, c0)
            b1 = np.add(w1, c1)
            wwScore = 1 - spatial.distance.cosine(w0, w1)
            ccScore = 1 - spatial.distance.cosine(c0, c1)
            bbScore = 1 - spatial.distance.cosine(b0, b1)

            RW0W1 = [x for x, y in
                     enumerate(dictionaryW[word1])
                     if y[0] == word0]
            RW1W0 = [x for x, y
                     in enumerate(dictionaryW[word0])
                     if y[0] == word1]
            RC0C1 = [x for x, y in
                     enumerate(dictionaryC[word1])
                     if y[0] == word0]
            RC1C0 = [x for x, y in
                     enumerate(dictionaryC[word0])
                     if y[0] == word1]

            RW0W1 = (n + 1) if len(RW0W1) == 0 else RW0W1[0]
            RW1W0 = (n + 1) if len(RW1W0) == 0 else RW1W0[0]
            RC0C1 = (n + 1) if len(RC0C1) == 0 else RC0C1[0]
            RC1C0 = (n + 1) if len(RC1C0) == 0 else RC1C0[0]

            outf.write(word0 + "-" + word1
                       + "," + words[2]
                       + "," + str(wwScore)
                       + "," + str(ccScore)
                       + "," + str(bbScore)
                       + "," + str(RW0W1)
                       + "," + str(RW1W0)
                       + "," + str(RC0C1)
                       + "," + str(RC1C0)
                       + "\n")
            print("Current time: " + str(datetime.now().time()))
        else:
            print("One of these words missing in our model:",
                  word0,
                  word1)
    outf.close()
    return similarities


def experiment2(wordpairFile, modelWord, modelContext,  outputFile):
    print("FA: dictionary of cue-responses is being made")
    wordpairs = open(wordpairFile).read().splitlines()
    dictionary = dict()
    for index, pair in enumerate(wordpairs):
        words = pair.split(',')
        cue = words[0].lower()
        print(words)
        print("Cue: " + cue)
        newResponses = words[1].lower().split('#')
        print("Responses: " + str(newResponses))
        if (cue in dictionary):
            responses = dictionary.get(cue)
            responses = list(set().union(newResponses, responses))
            dictionary[cue] = responses
        else:
            dictionary[cue] = newResponses
        #print(dictionary)
    print("FA: guesses are being made...")
    outf = open(outputFile, 'w')
    mw = modelWord
    mc = modelContext

    # Cue words, the total number of predictions made by the model
    # and the number of right predictions
    outf.write("CueWord" + ","
               + "humanResponses" + ","
               + "modelResponsesW" + ","
               + "modelResponsesC" + ","
               + "totalGuesses" + ","
               + "trueGuessesWW" + ","
               + "trueGuesseWC" + "\n")
    result = ""
    for totalGuesses in [10, 20, 30, 40, 50, 100]:
        foundTotalW = 0
        foundTotalC = 0

        for cue in dictionary.keys():
            if cue not in mw.vocab:
                print("Cue word missing in our model:", cue)
                continue
            humanResponses = dictionary.get(cue)
            w0 = mw[cue]
            c0 = mc[cue]
            w0 = normalize(w0[:, np.newaxis], axis=0).ravel()
            c0 = normalize(c0[:, np.newaxis], axis=0).ravel()
            b0 = np.add(w0, c0)
            b0 = normalize(b0[:, np.newaxis], axis=0).ravel()
            modelResponsesW = mw.most_similar(cue, topn=totalGuesses+1)
            modelResponsesC = mc.most_similar(cue, topn=totalGuesses+1)

            responsesW = [x[0] for x in modelResponsesW[:totalGuesses - 1]]
            foundW = list(set(responsesW).intersection(humanResponses))
            responsesC = [x[0] for x in modelResponsesC[:totalGuesses - 1]]
            foundC = list(set(responsesC).intersection(humanResponses))
            foundTotalW += len(foundW)
            foundTotalC += len(foundC)
            outf.write(cue + ","
                       #+ "," + str(humanResponses)
                       #+ "," + str(modelResponsesW)
                       #+ "," + str(modelResponsesC)
                       + "," + str(totalGuesses)
                       + "," + str(len(foundW))
                       + "," + str(len(foundC))
                       + "\n")
            print("Cue: " + cue)
            print("Human responses:" + str(humanResponses))
            print("Model W guesses:" + str(responsesW))
            print("Model C guesses:" + str(responsesC))
            print("Model W found:" + str(foundW))
            print("Model C found:" + str(foundC))
            print("\nWith totalGuesses =", str(totalGuesses), ":")
            print("\tfoundTotalW =", str(foundTotalW))
            print("\tfoundTotalC =", str(foundTotalC))
    outf.close()
    print(result)
    return experiment2


def experiment1(modelRepository, inputPath, outputPath):
    ## model repository is where folders including pretrained vector spaces located. Each vector should include a vectorsW.txt and vectorsC.txt file.
    ## inputPath includes similarity and relatedness datasets (for ACL we used SimLex.Gold.csv and McRaeTotal-Gold.csv respectively) work on each dataset at a time by commenting out the other.
    ## outputPath is where the result file (numbers for filling the table in the paper) will be located.
    models = list()
    for dirName, dirNames, fileNames in os.walk(modelRepository):
        # print path to all filenames.
        for modelName in dirNames:
            print(os.path.join(dirName, modelName))
            models.append(os.path.join(dirName, modelName))
               
    for model in models:
        print("Current time: " + str(datetime.now().time()))
        print("Loading model from: " + model)
        modelW = model + "/vectorsW"
        modelC = model + "/vectorsC"
        print("Loading W model...")
        mw =  KeyedVectors.load_word2vec_format(modelW + ".txt", binary=False) 
        mw.save_word2vec_format(modelW + ".bin", binary=True) 
        print("Loading C model...")
        mc =  KeyedVectors.load_word2vec_format(modelC + ".txt", binary=False)
        mc.save_word2vec_format(modelC + ".bin", binary=True) 

        ## For faster in the future:
        ## model = word2vec.Word2Vec.load_word2vec_format('')
        ## model.save_word2vec_format('', binary=true)

        ## add features using distributional vectors
        similarities(inputPath + "SimLex-Gold.csv", mw, mc, outputPath + "SimLex-Features.csv")
        similarities(inputPath + "McRaeTotal-Gold.csv", mw, mc, outputPath + "McRaeTotal-Features.csv")
        outf= open(outputPath + "Result-" + model.replace("/","-") + ".txt" , 'w')
        
        
        ## use features for classification (use simlex data for similarity and mcrae for relatedness)
        print("**** SIMILARITY ****")
        dataframe = pd.read_csv(outputPath + "SimLex-Features.csv")
        print(dataframe.shape)
        allLabels = dataframe.GoldSimilarity
        allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW']]
        allFeatures = np.array(allFeatures)
        classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        classifier.fit(allFeatures, allLabels)
        print(classifier.score(allFeatures, allLabels))
        p = classifier.predict(allFeatures)	
        dataframe["predicted"] = p
        mystr = str(dataframe.head(20)) + "\n\n" + \
            "\n Correlation bw AllReg and gold:" + str(scipy.stats.spearmanr(p, allLabels)) +\
            "\n Correlation bw WW and gold:" + str(scipy.stats.spearmanr(dataframe.WW, allLabels).correlation) +\
            "\n Correlation bw CC and gold:" + str(scipy.stats.spearmanr(dataframe.CC, allLabels).correlation) +\
            "\n Correlation bw WC and gold:" + str(scipy.stats.spearmanr(dataframe.WC, allLabels).correlation) +\
            "\n Correlation bw CW and gold:" + str(scipy.stats.spearmanr(dataframe.CW, allLabels).correlation) # +\
            #"\n Correlation bw AA and gold:" + str(scipy.stats.spearmanr(dataframe.AA, allLabels).correlation)
        print(mystr)
        outf.write(mystr) 



        print("**** RELATEDNESS McRae ****")
        dataframe = pd.read_csv(outputPath + "McRaeTotal-Features.csv")
        #print dataframe.shape
        allLabels = dataframe.GoldSimilarity
        allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW']]
        allFeatures = np.array(allFeatures)
        classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        classifier.fit(allFeatures, allLabels)
        #print(classifier.score(allFeatures, allLabels))
        p = classifier.predict(allFeatures)	
        dataframe["predicted"] = p
        mystr = str(dataframe.head(20)) + "\n\n" + \
            "\n Correlation bw AllReg and gold:" + str(scipy.stats.spearmanr(p, allLabels)) +\
            "\n Correlation bw WW and gold:" + str(scipy.stats.spearmanr(dataframe.WW, allLabels).correlation) +\
            "\n Correlation bw CC and gold:" + str(scipy.stats.spearmanr(dataframe.CC, allLabels).correlation) +\
            "\n Correlation bw WC and gold:" + str(scipy.stats.spearmanr(dataframe.WC, allLabels).correlation) +\
            "\n Correlation bw CW and gold:" + str(scipy.stats.spearmanr(dataframe.CW, allLabels).correlation)# +\
            #"\n Correlation bw AA and gold:" + str(scipy.stats.spearmanr(dataframe.AA, allLabels).correlation)
        print(mystr)
        outf.write(mystr) 
        
        
        print("Current time: " + str(datetime.now().time()))       
        outf.close()
    return experiment1
    
 
import getopt         
           
def main(argv):
    modelRepository = "/Users/fa/workspace/repos/_codes/MODELS/Rob/"
    inputPath = "/Users/fa/workspace/repos/_codes/sharedTask/classification-data/input/"
    outputPath = "/Users/fa/workspace/repos/_codes/sharedTask/classification-data/output/"
    
    try:
        opts, args = getopt.getopt(argv,"hm:i:o:",["mrepos=","ifile=","ofile="])
        print(opts)
        print(args)
    except getopt.GetoptError:
        print('test.py -m <modelrepos> -i <inputfile> -o <outputfile(s)>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -m <modelrepos> -i <inputfile> -o <outputfile(s)>')
            sys.exit()
        elif opt in ("-m", "--mrepos"):
            modelRepository = arg
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            
    
    experiment1(modelRepository, inputPath , outputPath)
    

if __name__ == "__main__":
    #main(sys.argv[1:])
   

    ### Experiment 1
    #print("Current time: " + str(datetime.now().time()))
    inputPath = "classification-data/input/"
    outputPath = "classification-data/output/"
    #modelRepository = "/Users/fa/workspace/repos/_codes/MODELS/Rob/Test/"
    #experiment1(modelRepository, inputPath , outputPath)


    ### Experiment 2
    print("Current time: " + str(datetime.now().time()))
    modelRepository = "/data/wordvet/"
    modelW = KeyedVectors.load_word2vec_format(modelRepository + "vectorsW.txt", binary=False) 
    modelC = KeyedVectors.load_word2vec_format(modelRepository + "vectorsC.txt", binary=False)
    wordpairFile = "classification-data/input/McRaeList-Gold.csv"
    outputFile = "classification-data/output/McRaeList-GuessingEvaluation.txt"
    experiment1("/data/", inputPath , outputPath)
    #experiment2(wordpairFile, modelW, modelC,  outputFile)

    
  
