*******************************************************************************************************
*                      			SemEval 2017 task 2                      		      *
* 			Multilingual and Cross-lingual Semantic Word Similarity        		      *
* 		José Camacho-Collados, Mohammad Taher Pilehvar, Nigel Collier, and Roberto Navigli    *
*******************************************************************************************************


This package contains the trial data for the SemEval 2017 task 2 on Multilingual and Cross-lingual Semantic Word Similarity. 

In addition to this README file, there is a directory containing the trial data ("trial") and a Java evaluation script ("task2-scorer.jar"):


The trial (directory) contains two folders for the two subtasks:

* subtask1-monolingual: contains five monolingual datasets for English, Farsi, German, Italian and Spanish.

* subtask2-crosslingual: contains ten cross-lingual datasets for all language pairs of the five languages mentioned above.


Each of these contain three folders:

** data
	
	- all the datasets, each line corresponding to a word pair: 
		word1 <tab> word2

		Language ISO codes: DE-German, EN-English, ES-Spanish, FA-Farsi (Persian), IT-Italian.


** keys

	- contains gold standard scores for all the datasets.


** output

	- contains sample system outputs for all the datasets.
	


"task2-scorer.jar": official evaluation script

	The official evaluation script for this task can be run from the terminal as follows:

		$ java -jar task2-scorer.jar [gold_key] [output_key]

	Example of usage:

		$ java -jar task2-scorer.jar trial/subtask1-monolingual/keys/de.trial.gold.txt trial/subtask1-monolingual/output/de.trial.sample.output.txt



For further information you can join our Google Group at https://groups.google.com/group/semeval-wordsimilarity) or contact us at:

José Camacho Collados (collados@di.uniroma1.it)
Mohammad Taher Pilehvar (mp792@cam.ac.uk)
