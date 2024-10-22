args = commandArgs(trailingOnly=TRUE)
if (length(args) != 3) {
  cat("trainGloveModel.R <size> <window> <iters>\n")
} else {

suppressMessages(library("text2vec"))
vectorSize   <- as.numeric(args[1])
window       <- as.numeric(args[2])
iters        <- as.numeric(args[3])

print("started running trainGloveModel.r")
text8_file = "data/wiki.shuffled-norm1-phrase1"

wiki = readLines(text8_file, warn = FALSE)
# Create iterator over tokens
tokens <- space_tokenizer(wiki)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)

vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, doc_proportion_max = 0.1,
                          vocab_term_max = 100000L, term_count_min = 1L)
# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)

tcm <- create_tcm(it, vectorizer, skip_grams_window=window)

ind <- tcm@x >= 1
tcm@x <- tcm@x[ind]
tcm@i <- tcm@i[ind]
tcm@j <- tcm@j[ind]

glove = GlobalVectors$new(word_vectors_size = vectorSize, vocabulary = vocab, x_max = 10)
glove$fit_transform(tcm, n_iter = iters)


dd <- glove$dump()
wordVec = dd$w_i
contextVec = dd$w_j
listVocab = vocab$term

W = cbind(listVocab, wordVec)
modelDir <- paste("models/glove_", vectorSize, "_", window, sep="")
dir.create(modelDir, showWarnings=FALSE)

vectorsW <- paste(modelDir, "/vectorsW.txt", sep="")
fileConn <- file(vectorsW)
writeLines(paste(nrow(wordVec), ncol(wordVec)),fileConn )
close(fileConn)
write.table(W, file = vectorsW, append = TRUE, quote = FALSE, sep = " ",
            eol = "\n", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

vectorsC <- paste(modelDir, "/vectorsC.txt", sep="")
C = cbind(listVocab, contextVec)
fileConn < -file(vectorsC)
writeLines(paste(nrow(contextVec), ncol(contextVec)),fileConn )
close(fileConn)
write.table(C, file = vectorsC, append = TRUE, quote = FALSE, sep = " ",
            eol = "\n", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

}
