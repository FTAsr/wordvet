
##pre-requisites:
#library(devtools)
#load_all("text2vec")
#install("text2vec")
#build("text2vec")


print("started running trainGloveModel.r")

library("text2vec")

text8_file = "/Users/fa/workspace/repos/_codes/data/text8"
wiki = readLines(text8_file, n = 1, warn = FALSE)
# Create iterator over tokens
tokens <- space_tokenizer(wiki)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 5L)
# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab) 
                               
tcm <- create_tcm(it, vectorizer)
glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
glove$fit(tcm, n_iter = 1)

dd <- glove$dump_model()
wordVec = dd$w_i
contextVec = dd$w_j
listVocab = vocab$vocab[,1]

W = cbind(listVocab, wordVec)
fileConn<-file("vectorsW.txt")
writeLines(paste(nrow(wordVec), ncol(wordVec)),fileConn )
close(fileConn)
write.table(W, file = "vectorsW.txt", append = TRUE, quote = FALSE, sep = " ",
            eol = "\n", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

C = cbind(listVocab, contextVec)           
fileConn<-file("vectorsC.txt")
writeLines(paste(nrow(contextVec), ncol(contextVec)),fileConn )
close(fileConn)
write.table(C, file = "vectorsC.txt", append = TRUE, quote = FALSE, sep = " ",
            eol = "\n", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

