# HW 1 NLP 517
# Wes Saunders -- All rights reserved etc

# Reads a sentence from our file and prints the test data
from files.PerplexityScorer import calculate_perplexity
from unigrammodel import *

SMALL_BROWN_TRAIN_TXT = "./files/CSEP517-HW1-Data-Small/brown.train.txt"
SMALL_BROWN_DEV_TXT = "./files/CSEP517-HW1-Data-Small/brown.dev.txt"


if __name__ == '__main__':
    unigramModel = UnigramModel(SMALL_BROWN_TRAIN_TXT)
    perplexity = calculate_perplexity(SMALL_BROWN_DEV_TXT, unigramModel.probabilities)
    print("Perplexity Score for brown.dev -- unigram model (trained on brown.train) \n \t PERPLEXITY: {}".format(perplexity))
    perplexity = calculate_perplexity(SMALL_BROWN_TRAIN_TXT, unigramModel.probabilities)
    print("Perplexity Score for brown.train -- unigram model (trained on brown.train) \n \t PERPLEXITY: {}".format(perplexity))