
##pre-requisites:
#library(devtools)
#load_all("text2vec")
#install("text2vec")
#build("text2vec")


print("started running trainGloveModel.r")

library("text2vec")

text8_file = "/data/wiki.shuffled-norm1-phrase1"

wiki = readLines(text8_file, n = 10000000, warn = FALSE)
# Create iterator over tokens
tokens <- space_tokenizer(wiki)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)

vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, doc_proportion_max = 0.3, 
                          max_number_of_terms = 30000L, term_count_min = 5L)
# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab, skip_grams_window=10) 
                               
tcm <- create_tcm(it, vectorizer)

ind <- tcm@x >= 1
tcm@x <- tcm@x[ind]
tcm@i <- tcm@i[ind]
tcm@j <- tcm@j[ind]

glove = GlobalVectors$new(word_vectors_size = 300, vocabulary = vocab, x_max = 10)
glove$fit(tcm, n_iter = 10)

dd <- glove$dump_model()
wordVec = dd$w_i
contextVec = dd$w_j
listVocab = vocab$vocab[,1]

W = cbind(listVocab, wordVec)
fileConn<-file("/data/model_200_10/vectorsW.txt")
writeLines(paste(nrow(wordVec), ncol(wordVec)),fileConn )
close(fileConn)
write.table(W, file = "/data/model_200_10/vectorsW.txt", append = TRUE, quote = FALSE, sep = " ",
            eol = "\n", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

C = cbind(listVocab, contextVec)           
fileConn<-file("/data/model_200_10/vectorsC.txt")
writeLines(paste(nrow(contextVec), ncol(contextVec)),fileConn )
close(fileConn)
write.table(C, file = "/data/model_200_10/vectorsC.txt", append = TRUE, quote = FALSE, sep = " ",
            eol = "\n", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

