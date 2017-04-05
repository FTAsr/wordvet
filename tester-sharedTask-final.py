#!/Users/fa/anaconda/bin/python

    
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import matplotlib.pyplot as plt
import numpy as np
import bisect 
import math

 
import multiprocessing
import openpyxl 
import os
import sys
import subprocess
import docopt
import datetime


from scipy import spatial
from sklearn.preprocessing import normalize

import pandas as pd
from sklearn import svm
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn import metrics as mt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import scipy

'''''
def similarities1(wordpairFile, goldScoreFile, modelFiles,  outputFile):
    outf = open(outputFile, 'w')
    models = list()
    for modelFile in modelFiles:
        print(modelFile)
        models.append ( Word2Vec.load_word2vec_format(modelFile, binary=False) ) # C text format
    print("FA: similarities!")
   
    wordpairs = [line.rstrip('\n') for line in open(wordpairFile)]
    goldScores = [line.rstrip('\n') for line in open(goldScoreFile)]
    for index, pair in enumerate(wordpairs):
        words = pair.split('\t')
        print(words)
        outf.write( words[0] + "\t" + word1 + "\t" + goldScores[index] + "\t")
        for model in models:
            if(words[0] in model.vocab and word1 in model.vocab):
                score = ( model.similarity(words[0], word1) * 2.0 ) + 2
            else:
                score = 0
            outf.write( str(score) + "\t" )
        outf.write( str(score) + "\n" )
    outf.close()
    return similarities
    
def test(wordpairFile, goldScoreFile, modelRepository, outputFile):
    modelFiles = list()
    for dirName, dirNames, fileNames in os.walk(modelRepository):
        # print path to all filenames.
        for fileName in fileNames:
            if "DS_Store" not in fileName:
                print(os.path.join(dirName, fileName))
                modelFiles.append(os.path.join(dirName, fileName))
    
    similarities1(wordpairFile, goldScoreFile, modelFiles, outputFile)
    return 0
'''   
    

def similarities(wordpairFile, modelWord, modelContext,  outputFile, formula = "all"):
    outf = open(outputFile, 'w')
    mw = modelWord
    mc = modelContext
    print("FA: similarity calculation with formula: " + formula)
    
    outf.write( "Wordpair" + "," + "GoldSimilarity" # word pair and the gold similarity score
                 + "," + "WW" 
                 + "," + "CC" 
                 + "," + "BB" 
                 + "," + "WC" 
                 + "," + "CW" + "\n" )
                 
                 
    #wordpairs = [line.rstrip('\n') for line in open(wordpairFile)]
    wordpairs = open(wordpairFile).read().splitlines()
    for index, pair in enumerate(wordpairs):
        words = pair.split(',')
        #print(words)
        score = 0.0
        word0 = words[0].lower()
        word1 = words[1].lower()
        word0 = word0.replace(" ","_")
        word1 = word1.replace(" ","_")
        word0 = word0.replace("-","_")
        word1 = word1.replace("-","_")
        if(word0 in mw.vocab and word1 in mw.vocab):
            w0 = mw[word0]
            w1 = mw[word1]
            c0 = mc[word0]
            c1 = mc[word1]
            w0 = normalize(w0[:,np.newaxis], axis=0).ravel()
            w1 = normalize(w1[:,np.newaxis], axis=0).ravel()
            c0 = normalize(c0[:,np.newaxis], axis=0).ravel()
            c1 = normalize(c1[:,np.newaxis], axis=0).ravel()
            
            b0 = np.add(w0 , c0)
            b1 = np.add(w1 , c1)
            
            b0 = normalize(b0[:,np.newaxis], axis=0).ravel()
            b1 = normalize(b1[:,np.newaxis], axis=0).ravel()
            
            wwScore = 1 - spatial.distance.cosine(w0,w1)
            ccScore = 1 - spatial.distance.cosine(c0,c1)
            wcScore = 1 - spatial.distance.cosine(w0, c1)
            cwScore = 1 - spatial.distance.cosine(w1, c0)
            bbScore = 1 - spatial.distance.cosine(b0, b1)
     
           
            outf.write( word0 + "-" + word1 + "," + words[2] # word pair and the gold similarity score
                     + "," + str(wwScore) 
                     + "," + str(ccScore) 
                     + "," + str(bbScore)
                     + "," + str(wcScore) 
                     + "," + str(cwScore) + "\n" )
        else:
            print("One of these words missing in our model: *" + word0 + "* *" + word1 + "*")
            outf.write( word0 + "-" + word1 + "," + words[2] # word pair and the gold similarity score
                     + "," + str(0.0) 
                     + "," + str(0.0) 
                     + "," + str(0.0)
                     + "," + str(0.0) 
                     + "," + str(0.0) + "\n" )
    outf.close()
    return similarities
    
    
       
    
def train(textRepository, modelRepository):
    # Trains word2vec model on text files from textRepository
    # Creates vectorsW.txt and vectorsC.txt in modelRepository    
    return 0



def test(modelRepository, inputPath , outputPath):
    # Uses the models in the modelRepository for similarity inference (different formulas)
    # Creates similarity score files in the outputPath
    
    modelW = modelRepository + "vectorsW.txt"
    modelC = modelRepository + "vectorsC.txt"
    print("Loading models...")
    mw =  Word2Vec.load_word2vec_format(modelW, binary=False) 
    print("Loading models...")
    mc =  Word2Vec.load_word2vec_format(modelC, binary=False)
    
    ## For faster in the future:
    ## model = word2vec.Word2Vec.load_word2vec_format('')
    ## model.save_word2vec_format('', binary=true)
   
   
    ### simple formulative approach
    #for formula in ["ww","cc","wc","cw","bb", "v3"]:
    #    similarities(inputPath, mw, mc, outputPath + formula + ".txt" , formula)
    
    ### supervised classification approach using simlex data
    ## add features using distributional vectors:
    similarities(inputPath + "SimLex-Gold-Nouns.csv", mw, mc, outputPath + "SimLex-Features-Nouns.csv")
    #similarities(inputPath + "Association-Gold.csv", mw, mc, outputPath + "Association-Features.csv")
    similarities(inputPath + "SharedTask-Gold.csv", mw, mc, outputPath + "SharedTask-Features.csv")
    ## use features for classification (use simlex data for traing and cross-validation of a svm model)
    dataframe = pd.read_csv(outputPath + "SimLex-Features-Nouns.csv")
    print dataframe.shape
    allLabels = dataframe.GoldSimilarity
    allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW','BB']]
    allFeatures = np.array(allFeatures)
    
    
    lm = smf.ols(formula='GoldSimilarity ~ WW * CC  ', data = dataframe).fit()
    print(lm.params)
    print(lm.pvalues)
    print(lm.summary()) 
    classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    classifier.fit(allFeatures, allLabels)
    print(classifier.score(allFeatures, allLabels))
    p = classifier.predict(allFeatures)	
    dataframe["predicted"] = p
    print(dataframe.head(20))
    print("Correlation bw predicted and gold:")
    print(scipy.stats.pearsonr(p, allLabels))
    print("Correlation bw WW and gold:")
    print(scipy.stats.pearsonr(dataframe.WW, allLabels))
    print("Correlation bw CC and gold:")
    print(scipy.stats.pearsonr(dataframe.CC, allLabels))
    print("Correlation bw BB and gold:")
    print(scipy.stats.pearsonr(dataframe.BB, allLabels))
    
    ## On trial data
    dataframe2 = pd.read_csv(outputPath + "SharedTask-Features.csv")
    print dataframe2.shape
    allLabels2 = dataframe2.GoldSimilarity
    allFeatures2 = dataframe2.ix[:,['WW', 'CC','WC','CW','BB']]
    allFeatures2 = np.array(allFeatures2)
    p = classifier.predict(allFeatures2)	
    dataframe2["predicted"] = p
    print(dataframe2.head(20))
    print("Correlation bw predicted and gold:")
    print(scipy.stats.pearsonr(p, allLabels2))
    print("Correlation bw WW and gold:")
    print(scipy.stats.pearsonr(dataframe2.WW, allLabels2))
    print("Correlation bw CC and gold:")
    print(scipy.stats.pearsonr(dataframe2.CC, allLabels2))
    print("Correlation bw BB and gold:")
    print(scipy.stats.pearsonr(dataframe2.BB, allLabels2))
    
    
    ''''
    
    classifier = SVR(C=1.0, epsilon=0.2)
    classifier.fit(allFeatures, allLabels)
    p = classifier.predict(allFeatures)	
    dataframe["predicted"] = p
    print(dataframe)
    
    
    kfold = cross_validation.KFold(len(allFeatures), n_folds=5)
    print("\nResults:")
    result = [classifier.fit(allFeatures[train], allLabels[train]).score(allFeatures[test], allLabels[test])  for train, test in kfold]
    print(result)
    print("\tAccuracy: %0.2f (+/- %0.2f)" % (np.mean(result), np.std(result) * 2))

    '''
    
import getopt         
           
def main(argv):
    #modelRepository = "/Users/fa/workspace/repos/_codes/MODELS/text8exp/-cbow0-size100-window5-negative5-hs1-sample0-threads10-binary0-iter1/"
    modelRepository = "/Users/fa/workspace/repos/_codes/MODELS/Rob/word2vec100_6/"
    inputPath = "/Users/fa/workspace/repos/_codes/sharedTask/classification-data/input/"
    outputPath = "/Users/fa/workspace/repos/_codes/sharedTask/classification-data/output/"
    
    try:
        opts, args = getopt.getopt(argv,"hm:i:o:",["mrepos=","ifile=","ofile="])
        print(opts)
        print(args)
    except getopt.GetoptError:
        print 'test.py -m <modelrepos> -i <inputfile> -o <outputfile(s)>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -m <modelrepos> -i <inputfile> -o <outputfile(s)>'
            sys.exit()
        elif opt in ("-m", "--mrepos"):
            modelRepository = arg
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    test(modelRepository, inputPath , outputPath)
    

if __name__ == "__main__":
#   main(sys.argv[1:])
   

###OUT OF FUNCTION: FOR COMMANDLINE


modelRepository = "/Users/fa/workspace/repos/_codes/MODELS/Rob/phrase/word2vec_200_12/"
inputPath = "/Users/fa/workspace/FA23/wordvet/classification-data/input/"
outputPath = "/Users/fa/workspace/FA23/wordvet/classification-data/output/"


modelW = modelRepository + "vectorsW"
modelC = modelRepository + "vectorsC"
print("Loading models...")
mw =  Word2Vec.load_word2vec_format(modelW + ".txt", binary=False) 
mw.save_word2vec_format(modelW + ".bin", binary=True) 
print("Loading models...")
mc =  Word2Vec.load_word2vec_format(modelC + ".txt", binary=False)
mc.save_word2vec_format(modelC + ".bin", binary=True) 

## For faster in the future:
## model = word2vec.Word2Vec.load_word2vec_format('')
## model.save_word2vec_format('', binary=true)


### simple formulative approach
#for formula in ["ww","cc","wc","cw","bb", "v3"]:
#    similarities(inputPath, mw, mc, outputPath + formula + ".txt" , formula)

### supervised classification approach using simlex data
## add features using distributional vectors:
similarities(inputPath + "SimLex-Gold.csv", mw, mc, outputPath + "SimLex-Features.csv")
similarities(inputPath + "Mix-Gold.csv", mw, mc, outputPath + "Mix-Features.csv")
#similarities(inputPath + "McRaeTotal-Gold.csv", mw, mc, outputPath + "McRaeTotal-Features.csv")
#similarities(inputPath + "Association-Gold.csv", mw, mc, outputPath + "Association-Features.csv")
similarities(inputPath + "SharedTask-Gold.csv", mw, mc, outputPath + "SharedTask-Features.csv")
similarities(inputPath + "SharedTaskActual-Gold.csv", mw, mc, outputPath + "SharedTaskActual-Features.csv")

print("Training on Mix (Normalized SimLex and McRae)")

## use features for classification (use simlex data for traing and cross-validation of a svm model)
dataframe = pd.read_csv(outputPath + "Mix-Features.csv")
print dataframe.shape
allLabels = dataframe.GoldSimilarity
allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW','BB']]
allFeatures = np.array(allFeatures)
classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
classifier.fit(allFeatures, allLabels)
print(classifier.score(allFeatures, allLabels))
p = classifier.predict(allFeatures)	
dataframe["predicted"] = p
print(dataframe.head(20))
print("Correlation bw predicted and gold:")
print(scipy.stats.pearsonr(p, allLabels))
print(scipy.stats.spearmanr(p, allLabels))
print("Correlation bw WW and gold:")
print(scipy.stats.pearsonr(dataframe.WW, allLabels))
print(scipy.stats.spearmanr(dataframe.WW, allLabels))
print("Correlation bw CC and gold:")
print(scipy.stats.pearsonr(dataframe.CC, allLabels))
print(scipy.stats.spearmanr(dataframe.CC, allLabels))
print("Correlation bw BB and gold:")
print(scipy.stats.pearsonr(dataframe.BB, allLabels))
print(scipy.stats.spearmanr(dataframe.BB, allLabels))
print("Correlation bw WC and gold:")
print(scipy.stats.pearsonr(dataframe.WC, allLabels))
print(scipy.stats.spearmanr(dataframe.WC, allLabels))
print("Correlation bw CW and gold:")
print(scipy.stats.pearsonr(dataframe.CW, allLabels))
print(scipy.stats.spearmanr(dataframe.CW, allLabels))

print("Test on SharedTask Trial")

## test on trial data
dataframe2 = pd.read_csv(outputPath + "SharedTask-Features.csv")
print dataframe2.shape
allLabels2 = dataframe2.GoldSimilarity
allFeatures2 = dataframe2.ix[:,['WW', 'CC','WC','CW','BB']]
allFeatures2 = np.array(allFeatures2)
p = classifier.predict(allFeatures2)	
dataframe2["predicted"] = p
print(dataframe2.head(20))
print("Correlation bw predicted and gold:")
print(scipy.stats.pearsonr(p, allLabels2))
print(scipy.stats.spearmanr(p, allLabels2))
print("Correlation bw WW and gold:")
print(scipy.stats.pearsonr(dataframe2.WW, allLabels2))
print(scipy.stats.spearmanr(dataframe2.WW, allLabels2))
print("Correlation bw CC and gold:")
print(scipy.stats.pearsonr(dataframe2.CC, allLabels2))
print(scipy.stats.spearmanr(dataframe2.CC, allLabels2))
print("Correlation bw BB and gold:")
print(scipy.stats.pearsonr(dataframe2.BB, allLabels2))
print(scipy.stats.spearmanr(dataframe2.BB, allLabels2))
print("Correlation bw WC and gold:")
print(scipy.stats.pearsonr(dataframe2.WC, allLabels2))
print(scipy.stats.spearmanr(dataframe2.WC, allLabels2))
print("Correlation bw CW and gold:")
print(scipy.stats.pearsonr(dataframe2.CW, allLabels2))
print(scipy.stats.spearmanr(dataframe2.CW, allLabels2))

#print("Result for SharedTask Actual")

### On Actual data
#dataframe2 = pd.read_csv(outputPath + "SharedTaskActual-Features.csv")
#print dataframe2.shape
#allFeatures2 = dataframe2.ix[:,['WW', 'CC','WC','CW','BB']]
#allFeatures2 = np.array(allFeatures2)
#p = classifier.predict(allFeatures2)	
#dataframe2["predicted"] = p
#dataframe2.to_csv(outputPath + "SharedTaskActual-ResultByTrainingOnMix.csv", encoding='utf-8')



print("Test on SharedTask Test (after release of final data)")

## test on trial data
dataframe2 = pd.read_csv(outputPath + "SharedTaskActual-Features_.csv")
print dataframe2.shape
allLabels2 = dataframe2.GoldSimilarity
allFeatures2 = dataframe2.ix[:,['WW', 'CC','WC','CW','BB']]
allFeatures2 = np.array(allFeatures2)
p = classifier.predict(allFeatures2)	
dataframe2["predicted"] = p
print(dataframe2.head(20))
print("Correlation bw predicted and gold:")
print(scipy.stats.pearsonr(p, allLabels2))
print(scipy.stats.spearmanr(p, allLabels2))
print("Correlation bw WW and gold:")
print(scipy.stats.pearsonr(dataframe2.WW, allLabels2))
print(scipy.stats.spearmanr(dataframe2.WW, allLabels2))
print("Correlation bw CC and gold:")
print(scipy.stats.pearsonr(dataframe2.CC, allLabels2))
print(scipy.stats.spearmanr(dataframe2.CC, allLabels2))
print("Correlation bw BB and gold:")
print(scipy.stats.pearsonr(dataframe2.BB, allLabels2))
print(scipy.stats.spearmanr(dataframe2.BB, allLabels2))
print("Correlation bw WC and gold:")
print(scipy.stats.pearsonr(dataframe2.WC, allLabels2))
print(scipy.stats.spearmanr(dataframe2.WC, allLabels2))
print("Correlation bw CW and gold:")
print(scipy.stats.pearsonr(dataframe2.CW, allLabels2))
print(scipy.stats.spearmanr(dataframe2.CW, allLabels2))















print("Training on only SimLex ")

## use features for classification (use simlex data for traing and cross-validation of a svm model)
dataframe = pd.read_csv(outputPath + "SimLex-Features.csv")
print dataframe.shape
allLabels = dataframe.GoldSimilarity
allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW','BB']]
allFeatures = np.array(allFeatures)
classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
classifier.fit(allFeatures, allLabels)
print(classifier.score(allFeatures, allLabels))
p = classifier.predict(allFeatures)	
dataframe["predicted"] = p
print(dataframe.head(20))
print("Correlation bw predicted and gold:")
print(scipy.stats.pearsonr(p, allLabels))
print(scipy.stats.spearmanr(p, allLabels))
print("Correlation bw WW and gold:")
print(scipy.stats.pearsonr(dataframe.WW, allLabels))
print(scipy.stats.spearmanr(dataframe.WW, allLabels))
print("Correlation bw CC and gold:")
print(scipy.stats.pearsonr(dataframe.CC, allLabels))
print(scipy.stats.spearmanr(dataframe.CC, allLabels))
print("Correlation bw BB and gold:")
print(scipy.stats.pearsonr(dataframe.BB, allLabels))
print(scipy.stats.spearmanr(dataframe.BB, allLabels))
print("Correlation bw WC and gold:")
print(scipy.stats.pearsonr(dataframe.WC, allLabels))
print(scipy.stats.spearmanr(dataframe.WC, allLabels))
print("Correlation bw CW and gold:")
print(scipy.stats.pearsonr(dataframe.CW, allLabels))
print(scipy.stats.spearmanr(dataframe.CW, allLabels))

print("Test on SharedTask Trial")

## test on trial data
dataframe2 = pd.read_csv(outputPath + "SharedTask-Features.csv")
print dataframe2.shape
allLabels2 = dataframe2.GoldSimilarity
allFeatures2 = dataframe2.ix[:,['WW', 'CC','WC','CW','BB']]
allFeatures2 = np.array(allFeatures2)
p = classifier.predict(allFeatures2)	
dataframe2["predicted"] = p
print(dataframe2.head(20))
print("Correlation bw predicted and gold:")
print(scipy.stats.pearsonr(p, allLabels2))
print(scipy.stats.spearmanr(p, allLabels2))
print("Correlation bw WW and gold:")
print(scipy.stats.pearsonr(dataframe2.WW, allLabels2))
print(scipy.stats.spearmanr(dataframe2.WW, allLabels2))
print("Correlation bw CC and gold:")
print(scipy.stats.pearsonr(dataframe2.CC, allLabels2))
print(scipy.stats.spearmanr(dataframe2.CC, allLabels2))
print("Correlation bw BB and gold:")
print(scipy.stats.pearsonr(dataframe2.BB, allLabels2))
print(scipy.stats.spearmanr(dataframe2.BB, allLabels2))
print("Correlation bw WC and gold:")
print(scipy.stats.pearsonr(dataframe2.WC, allLabels2))
print(scipy.stats.spearmanr(dataframe2.WC, allLabels2))
print("Correlation bw CW and gold:")
print(scipy.stats.pearsonr(dataframe2.CW, allLabels2))
print(scipy.stats.spearmanr(dataframe2.CW, allLabels2))

print("Result for SharedTask Actual")

## On Actual data
dataframe2 = pd.read_csv(outputPath + "SharedTaskActual-Features.csv")
print dataframe2.shape
allFeatures2 = dataframe2.ix[:,['WW', 'CC','WC','CW','BB']]
allFeatures2 = np.array(allFeatures2)
p = classifier.predict(allFeatures2)	
dataframe2["predicted"] = p
dataframe2.to_csv(outputPath + "SharedTaskActual-ResultByTrainingOnSimLex.csv", encoding='utf-8')