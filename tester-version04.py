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

def similarities(wordpairFile, modelWord, modelContext,  outputFile):
    outf = open(outputFile, 'w')
    mw = modelWord
    mc = modelContext
    print("FA: similarity calculation..." )
    
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
        word0 = words[0].lower().replace(" ","_")
        word1 = words[1].lower().replace(" ","_")
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
            print("One of these words missing in our model: " + word0 + " " + word1)
    outf.close()
    return similarities


def rank(m, targetWord, cueVector, restrict_vocab=None):
    """
    Find the rank of targetWord in the vocabulary wrt. its similarity to the cue vector 
    """
    m.init_sims()
    mean = cueVector
    limited = m.syn0norm if restrict_vocab is None else m.syn0norm[:restrict_vocab]
    dists = np.dot(limited, mean)
    best = matutils.argsort(dists, reverse=True)
    result = [(m.index2word[sim], float(dists[sim])) for sim in best]
    #print result[0]
    return [x for x, y in enumerate(result) if y[0] == targetWord]
    
def similaritiesAndRanks(wordpairFile, modelWord, modelContext,  outputFile):
    outf = open(outputFile, 'w')
    mw = modelWord
    mc = modelContext
    print("FA: similarity calculation..." )
    
    outf.write( "Wordpair" + "," + "GoldSimilarity" # word pair and the gold similarity score
                 + "," + "WW" 
                 + "," + "CC" 
                 + "," + "BB" 
                 + "," + "WC" 
                 + "," + "CW" 
                 + "," + "RW0W1" 
                 + "," + "RW1W0" 
                 + "," + "RW0C1" 
                 + "," + "RW1C0" + "\n" )
              
                 
    #wordpairs = [line.rstrip('\n') for line in open(wordpairFile)]
    wordpairs = open(wordpairFile).read().splitlines()
    for index, pair in enumerate(wordpairs):
        words = pair.split(',')
        print(words)
        score = 0.0
        word0 = words[0].lower().replace(" ","_")
        word1 = words[1].lower().replace(" ","_")
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
            
            # is one word a high-rank similar neigbor of the other?
            RW0W1 = rank(mw, word1, w0) #we have the vector of word0 from model W (w0) and we lookup the rank of word1 in model W (with respect to its closeness to word0)
            RW1W0 = rank(mw, word0, w1) #we have the vector of word1 from model W (w1) and we lookup the rank of word0 in model W (with respect to its closeness to word1)
            # is one word a high-rank related word to the other?
            RW0C1 = rank(mc, word1, w0) #we have the vector of word0 from model W (w0) and we lookup the rank of word1 in model C (with respect to its closeness to context of word0)
            RW1C0 = rank(mc, word0, w1) #we have the vector of word1 from model W (w1) and we lookup the rank of word0 in model C (with respect to its closeness to context of word1)
            
            outf.write( word0 + "-" + word1 + "," + words[2] # word pair and the gold similarity score
                     + "," + str(wwScore) 
                     + "," + str(ccScore) 
                     + "," + str(bbScore)
                     + "," + str(wcScore) 
                     + "," + str(cwScore)
                     + "," + str(RW0W1) 
                     + "," + str(RW1W0)
                     + "," + str(RW0C1)
                     + "," + str(RW1C0) + "\n" )
        else:
            print("One of these words missing in our model: " + word0 + " " + word1)
    outf.close()
    return similarities


def nguesses(wordpairFile, modelWord, modelContext,  outputFile):

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
        if (dictionary.has_key(cue)):
            responses = dictionary.get(cue)
            responses = list(set().union(newResponses,responses))
            dictionary[cue] = responses
        else:
            dictionary[cue] = newResponses
        #print(dictionary)
    print("FA: guesses are being made...")         
    outf = open(outputFile, 'w')
    mw = modelWord
    mc = modelContext
    outf.write( "CueWord" + "," 
                + "humanResponses" + "," 
                + "modelResponsesW" + "," 
                + "modelResponsesC"  + "," 
                + "totalGuesses" + "," 
                + "trueGuessesWW" + "," 
                + "trueGuesseWC" + "\n" ) #Cue words, the total number of predictions made by the model and the number of right predictions      
    result = ""
    foundTotalW = 0
    foundTotalC = 0
    for cue in dictionary.keys():
        if (cue not in mw.vocab):
            print("Cue word missing in our model: " + cue )
            continue
        humanResponses = dictionary.get(cue)
        w0 = mw[cue]
        c0 = mc[cue]
        w0 = normalize(w0[:,np.newaxis], axis=0).ravel()
        c0 = normalize(c0[:,np.newaxis], axis=0).ravel()
        b0 = np.add(w0 , c0)
        b0 = normalize(b0[:,np.newaxis], axis=0).ravel()
        modelResponsesW = mw.most_similar(cue, topn=200)
        modelResponsesC = mc.most_similar(cue, topn=200)
        for totalGuesses in [10, 20, 30, 40, 50, 100]:
            responsesW = [x[0] for x in modelResponsesW[:totalGuesses -1]]
            foundW = list(set(responsesW).intersection(humanResponses)) 
            responsesC = [x[0] for x in modelResponsesC[:totalGuesses -1]]
            foundC = list(set(responsesC).intersection(humanResponses)) 
            foundTotalW += len(foundW)
            foundTotalC += len(foundC)
            outf.write( cue + "," 
                     #+ "," + str(humanResponses) 
                     #+ "," + str(modelResponsesW) 
                     #+ "," + str(modelResponsesC) 
                     + "," + str(totalGuesses)
                     + "," + str(len(foundW))
                     + "," + str(len(foundC)) + "\n" )
            print("Cue: " + cue)
            print("Human responses:" + str(humanResponses))
            print("Model W guesses:" + str(responsesW))
            print("Model C guesses:" + str(responsesC))
            print("Model W found:" + str(foundW))
            print("Model C found:" + str(foundC))
        result = result + "\tfoundTotalW = " + str(foundTotalW) + "\n"
        result = result + "\tfoundTotalC = " + str(foundTotalC) + "\n"
    outf.close()
    return result

    

    
    
def test(modelRepository, inputPath , outputPath):
    models = list()
    for dirName, dirNames, fileNames in os.walk(modelRepository):
        # print path to all filenames.
        for modelName in dirNames:
            print(os.path.join(dirName, modelName))
            models.append(os.path.join(dirName, modelName))
               
    for model in models:
        print("Loading model from: " + model)
        modelW = model + "/vectorsW"
        modelC = model + "/vectorsC"
        print("Loading W model...")
        mw =  Word2Vec.load_word2vec_format(modelW + ".txt", binary=False) 
        mw.save_word2vec_format(modelW + ".bin", binary=True) 
        print("Loading C model...")
        mc =  Word2Vec.load_word2vec_format(modelC + ".txt", binary=False)
        mc.save_word2vec_format(modelC + ".bin", binary=True) 

         ## For faster in the future:
         ## model = word2vec.Word2Vec.load_word2vec_format('')
         ## model.save_word2vec_format('', binary=true)

         ## add features using distributional vectors:
         #similarities(inputPath + "SimLex-Gold-Nouns.csv", mw, mc, outputPath + "SimLex-Features-Nouns.csv")
        similarities(inputPath + "SimLex-Gold.csv", mw, mc, outputPath + "SimLex-Features.csv")
        similarities(inputPath + "McRaeTotal-Gold.csv", mw, mc, outputPath + "McRaeTotal-Features.csv")
        #similarities(inputPath + "Association-Gold.csv", mw, mc, outputPath + "Association-Features.csv")
        #similarities(inputPath + "SharedTask-Gold.csv", mw, mc, outputPath + "SharedTask-Features.csv")
        outf = open(outputPath + "Result-" + model.replace("/","-") + ".txt" , 'w')
        ## use features for classification (use simlex data for similarity and mcrae for relatedness)

        print("**** SIMILARITY ****")
        dataframe = pd.read_csv(outputPath + "SimLex-Features.csv")
        #print dataframe.shape
        allLabels = dataframe.GoldSimilarity
        allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW','BB']]
        allFeatures = np.array(allFeatures)
        classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        classifier.fit(allFeatures, allLabels)
        #print(classifier.score(allFeatures, allLabels))
        p = classifier.predict(allFeatures)	
        dataframe["predicted"] = p
        mystr = str(dataframe.head(20)) + "\n\n" + \
            "\n Correlation bw predicted and gold:" + str(scipy.stats.spearmanr(p, allLabels)) +\
            "\n Correlation bw WW and gold:" + str(scipy.stats.spearmanr(dataframe.WW, allLabels).correlation) +\
            "\n Correlation bw CC and gold:" + str(scipy.stats.spearmanr(dataframe.CC, allLabels).correlation) +\
            "\n Correlation bw WC and gold:" + str(scipy.stats.spearmanr(dataframe.WC, allLabels).correlation) +\
            "\n Correlation bw CW and gold:" + str(scipy.stats.spearmanr(dataframe.CW, allLabels).correlation) +\
            "\n Correlation bw BB and gold:" + str(scipy.stats.spearmanr(dataframe.BB, allLabels).correlation)
        #print(mystr)
        outf.write(mystr) 

        print("**** RELATEDNESS ****")
        dataframe = pd.read_csv(outputPath + "McRaeTotal-Features.csv")
        #print dataframe.shape
        allLabels = dataframe.GoldSimilarity
        allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW','BB']]
        allFeatures = np.array(allFeatures)
        classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        classifier.fit(allFeatures, allLabels)
        #print(classifier.score(allFeatures, allLabels))
        p = classifier.predict(allFeatures)	
        dataframe["predicted"] = p
        mystr = str(dataframe.head(20)) + "\n\n" + \
            "\n Correlation bw predicted and gold:" + str(scipy.stats.spearmanr(p, allLabels)) +\
            "\n Correlation bw WW and gold:" + str(scipy.stats.spearmanr(dataframe.WW, allLabels).correlation) +\
            "\n Correlation bw CC and gold:" + str(scipy.stats.spearmanr(dataframe.CC, allLabels).correlation) +\
            "\n Correlation bw WC and gold:" + str(scipy.stats.spearmanr(dataframe.WC, allLabels).correlation) +\
            "\n Correlation bw CW and gold:" + str(scipy.stats.spearmanr(dataframe.CW, allLabels).correlation) +\
            "\n Correlation bw BB and gold:" + str(scipy.stats.spearmanr(dataframe.BB, allLabels).correlation)
        #print(mystr)
        outf.write(mystr) 
        
        
        print("**** RELATEDNESS GUESSING ****")
        mystr = nguesses(inputPath + "McRaeList-Gold.csv", mw, mc,  outputPath + "McRaeList-Guesses.csv")
        outf.write(mystr)
        
        outf.close()
    
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
    #main(sys.argv[1:])
   

    ###OUT OF FUNCTION: FOR COMMANDLINE

    inputPath = "/Users/fa/workspace/FA23/wordvet/classification-data/input/"
    outputPath = "/Users/fa/workspace/FA23/wordvet/classification-data/output/"
    modelRepository = "/Users/fa/workspace/repos/_codes/MODELS/Rob/Test/"
    test(modelRepository, inputPath , outputPath)

    modelW = modelRepository + "vectorsW"
    modelC = modelRepository + "vectorsC"
    print("Loading models...")
    mw =  Word2Vec.load_word2vec_format(modelW + ".txt", binary=False) 
    print("Loading models...")
    mc =  Word2Vec.load_word2vec_format(modelC + ".txt", binary=False)

    similaritiesAndRanks(inputPath + "McRaeTotal-Gold.csv", mw, mc,  outputPath + "McRaeTotal-ExtraFeatures.csv")
    similaritiesAndRanks(inputPath + "SimLex-Gold.csv", mw, mc,  outputPath + "SimLex-ExtraFeatures.csv")
    
    print("**** SIMILARITY ****")
    dataframe = pd.read_csv(outputPath + "SimLex-ExtraFeatures.csv")
    #print dataframe.shape
    allLabels = dataframe.GoldSimilarity
    allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW','BB','R',"RW0W1", "RW1W0", "RW0C1", "RW1C0"]]#dataframe.ix[:,['WW', 'CC','WC','CW','BB']]
    allFeatures = np.array(allFeatures)
    classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    classifier.fit(allFeatures, allLabels)
    #print(classifier.score(allFeatures, allLabels))
    p = classifier.predict(allFeatures)	
    dataframe["predicted"] = p
    mystr = str(dataframe.head(20)) + "\n\n" + \
        "\n Correlation bw predicted and gold:" + str(scipy.stats.spearmanr(p, allLabels)) +\
        "\n Correlation bw WW and gold:" + str(scipy.stats.spearmanr(dataframe.WW, allLabels).correlation) +\
        "\n Correlation bw CC and gold:" + str(scipy.stats.spearmanr(dataframe.CC, allLabels).correlation) +\
        "\n Correlation bw WC and gold:" + str(scipy.stats.spearmanr(dataframe.WC, allLabels).correlation) +\
        "\n Correlation bw CW and gold:" + str(scipy.stats.spearmanr(dataframe.CW, allLabels).correlation) +\
        "\n Correlation bw BB and gold:" + str(scipy.stats.spearmanr(dataframe.BB, allLabels).correlation)
    print(mystr)
        
    print("**** RELATEDNESS ****")
    dataframe = pd.read_csv(outputPath + "McRaeTotal-ExtraFeatures.csv")
    #print dataframe.shape
    allLabels = dataframe.GoldSimilarity
    allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW','BB','R',"RW0W1", "RW1W0", "RW0C1", "RW1C0"]]#dataframe.ix[:,['WW', 'CC','WC','CW','BB']]
    
    allFeatures = np.array(allFeatures)
    classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    classifier.fit(allFeatures, allLabels)
    #print(classifier.score(allFeatures, allLabels))
    p = classifier.predict(allFeatures)	
    dataframe["predicted"] = p
    mystr = str(dataframe.head(20)) + "\n\n" + \
        "\n Correlation bw predicted and gold:" + str(scipy.stats.spearmanr(p, allLabels)) +\
        "\n Correlation bw WW and gold:" + str(scipy.stats.spearmanr(dataframe.WW, allLabels).correlation) +\
        "\n Correlation bw CC and gold:" + str(scipy.stats.spearmanr(dataframe.CC, allLabels).correlation) +\
        "\n Correlation bw WC and gold:" + str(scipy.stats.spearmanr(dataframe.WC, allLabels).correlation) +\
        "\n Correlation bw CW and gold:" + str(scipy.stats.spearmanr(dataframe.CW, allLabels).correlation) +\
        "\n Correlation bw BB and gold:" + str(scipy.stats.spearmanr(dataframe.BB, allLabels).correlation)
    print(mystr)


'''''

inputPath = "/Users/fa/workspace/FA23/wordvet/classification-data/input/"
outputPath = "/Users/fa/workspace/FA23/wordvet/classification-data/output/"
modelRepository = "/Users/fa/workspace/repos/_codes/MODELS/Rob/Test/"

modelW = modelRepository + "vectorsW"
modelC = modelRepository + "vectorsC"
print("Loading models...")
mw =  Word2Vec.load_word2vec_format(modelW + ".txt", binary=False) 
print("Loading models...")
mc =  Word2Vec.load_word2vec_format(modelC + ".txt", binary=False)



print("FA: dictionary of cue-responses is being made")     
wordpairs = open(wordpairFile).read().splitlines()
dictionary = dict()
for index, pair in enumerate(wordpairs):
    words = pair.split(',')
    cue = words[0].lower()
    print(words)
    print("Cue: " + cue)
    newResponses = words[1].lower().split('#')
    print("Responses: " + cue)
    if (dictionary.has_key(cue)):
        responses = dictionary.get(cue)
        responses = list(set().union(newResponses,responses))
        dictionary[cue] = responses
    else:
        dictionary[cue] = newResponses
    #print(dictionary)
print("FA: guesses are being made...")         
outf = open(outputFile, 'w')
mw = modelWord
mc = modelContext
outf.write( "CueWord" + "," 
            + "humanResponses" + "," 
            + "totalGuesses" + "," 
            + "trueGuessesWW" + "," 
            + "trueGuesseWC" + "\n" ) #Cue words, the total number of predictions made by the model and the number of right predictions      
result = ""
foundTotalW = 0
foundTotalC = 0
for cue in dictionary.keys():
    if (cue not in mw.vocab):
        print("Cue word missing in our model: " + cue )
        continue
    humanResponses = dictionary.get(cue)
    w0 = mw[cue]
    c0 = mc[cue]
    w0 = normalize(w0[:,np.newaxis], axis=0).ravel()
    c0 = normalize(c0[:,np.newaxis], axis=0).ravel()
    b0 = np.add(w0 , c0)
    b0 = normalize(b0[:,np.newaxis], axis=0).ravel()
    modelResponsesW = mw.most_similar(cue, topn=60)
    modelResponsesC = mc.most_similar(cue, topn=60)
    for totalGuesses in [10]:
        m = [x[0] for x in modelResponsesW[:totalGuesses]]
        foundW = list(set().intersection(m,humanResponses)) 
        m = [x[0] for x in modelResponsesC[:totalGuesses]]
        foundC = list(set().intersection(m,humanResponses)) 
        foundTotalW += len(foundW)
        foundTotalC += len(foundC)
        outf.write( cue + "," 
                 + "," + str(humanResponses) 
                 #+ "," + str(modelResponsesW) 
                 #+ "," + str(modelResponsesC) 
                 + "," + str(totalGuesses)
                 + "," + str(len(foundW))
                 + "," + str(len(foundC)) + "\n" )
        print("Cue: " + cue)
        print("Human responses:" + str(humanResponses))
        print("Model W found:" + str(modelResponsesW))
        print("Model C found:" + str(modelResponsesC))
        
    result = result + "\tfoundTotalW = " + str(foundTotalW) + "\n"
    result = result + "\tfoundTotalC = " + str(foundTotalC) + "\n"
outf.close()


    
    
    
    
    
    
models = list()
for dirName, dirNames, fileNames in os.walk(modelRepository):
    # print path to all filenames.
    for modelName in dirNames:
        print(os.path.join(dirName, modelName))
        models.append(os.path.join(dirName, modelName))
               
for model in models:

    modelW = model + "vectorsW"
    modelC = model + "vectorsC"
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
    #similarities(inputPath + "SimLex-Gold-Nouns.csv", mw, mc, outputPath + "SimLex-Features-Nouns.csv")
    similarities(inputPath + "SimLex-Gold.csv", mw, mc, outputPath + "SimLex-Features.csv")
    similarities(inputPath + "McRaeTotal-Gold.csv", mw, mc, outputPath + "McRaeTotal-Features.csv")
    #similarities(inputPath + "Association-Gold.csv", mw, mc, outputPath + "Association-Features.csv")
    #similarities(inputPath + "SharedTask-Gold.csv", mw, mc, outputPath + "SharedTask-Features.csv")
    outf = open(outputPath + "Result-" + model.replace("/","-") + ".txt" , 'w')
    ## use features for classification (use simlex data for similarity and mcrae for relatedness)

    print("**** SIMILARITY ****")
    dataframe = pd.read_csv(outputPath + "SimLex-Features.csv")
    #print dataframe.shape
    allLabels = dataframe.GoldSimilarity
    allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW','BB']]
    allFeatures = np.array(allFeatures)
    classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    classifier.fit(allFeatures, allLabels)
    #print(classifier.score(allFeatures, allLabels))
    p = classifier.predict(allFeatures)	
    dataframe["predicted"] = p
    mystr = str(dataframe.head(20)) + "\n\n" + \
        "\n Correlation bw predicted and gold:" + str(scipy.stats.spearmanr(p, allLabels)) +\
        "\n Correlation bw WW and gold:" + str(scipy.stats.spearmanr(dataframe.WW, allLabels).correlation) +\
        "\n Correlation bw CC and gold:" + str(scipy.stats.spearmanr(dataframe.CC, allLabels).correlation) +\
        "\n Correlation bw WC and gold:" + str(scipy.stats.spearmanr(dataframe.WC, allLabels).correlation) +\
        "\n Correlation bw CW and gold:" + str(scipy.stats.spearmanr(dataframe.CW, allLabels).correlation) +\
        "\n Correlation bw BB and gold:" + str(scipy.stats.spearmanr(dataframe.BB, allLabels).correlation)
    print(mystr)
    outf.write(mystr) 

    print("**** RELATEDNESS ****")
    dataframe = pd.read_csv(outputPath + "McRaeTotal-Features.csv")
    #print dataframe.shape
    allLabels = dataframe.GoldSimilarity
    allFeatures = dataframe.ix[:,['WW', 'CC','WC','CW','BB']]
    allFeatures = np.array(allFeatures)
    classifier = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    classifier.fit(allFeatures, allLabels)
    #print(classifier.score(allFeatures, allLabels))
    p = classifier.predict(allFeatures)	
    dataframe["predicted"] = p
    mystr = str(dataframe.head(20)) + "\n\n" + \
       "\n Correlation bw predicted and gold:" + str(scipy.stats.spearmanr(p, allLabels)) +\
       "\n Correlation bw WW and gold:" + str(scipy.stats.spearmanr(dataframe.WW, allLabels).correlation) +\
       "\n Correlation bw CC and gold:" + str(scipy.stats.spearmanr(dataframe.CC, allLabels).correlation) +\
       "\n Correlation bw WC and gold:" + str(scipy.stats.spearmanr(dataframe.WC, allLabels).correlation) +\
       "\n Correlation bw CW and gold:" + str(scipy.stats.spearmanr(dataframe.CW, allLabels).correlation) +\
       "\n Correlation bw BB and gold:" + str(scipy.stats.spearmanr(dataframe.BB, allLabels).correlation)
    print(mystr)
    outf.write(mystr) 
    outf.close()
    
    '''''