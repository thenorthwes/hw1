# HW 1 NLP 517
# Wes Saunders -- All rights reserved etc

# Reads a sentence from our file and prints the test data
from PerplexityScorer import calculate_perplexity, calculate_ngram_perplexity, calculate_linear_perplexity
from linear_interpolation import linear_interpolation
from ngrammodel import ngram
from unigrammodel import *

SMALL_BROWN_TRAIN_TXT = "./files/CSEP517-HW1-Data-Small/brown.train.txt"
SMALL_BROWN_DEV_TXT = "./files/CSEP517-HW1-Data-Small/brown.dev.txt"


if __name__ == '__main__':
    # unigramModel = UnigramModel(SMALL_BROWN_TRAIN_TXT)
    # perplexity = calculate_perplexity(SMALL_BROWN_DEV_TXT, unigramModel.probabilities)
    # print("Perplexity Score for brown.dev -- unigram model (trained on brown.train) \n \t PERPLEXITY: {}".format(
    #     perplexity))
    # perplexity = calculate_perplexity(SMALL_BROWN_TRAIN_TXT, unigramModel.probabilities)
    # print("Perplexity Score for brown.train -- unigram model (trained on brown.train) \n \t PERPLEXITY: {}".format(
    #     perplexity))
    #
    # # bigrams
    # bigram = ngram(SMALL_BROWN_TRAIN_TXT, 2)
    # perplexity = calculate_ngram_perplexity(SMALL_BROWN_DEV_TXT, bigram.vocabulary_space, bigram.probabilities, 2)
    # print("Perplexity Score for brown.dev -- bigram model (trained on brown.train) \n \t PERPLEXITY: {}".format(
    #     perplexity))
    # perplexity = calculate_ngram_perplexity(SMALL_BROWN_TRAIN_TXT, bigram.vocabulary_space, bigram.probabilities, 2)
    # print("Perplexity Score for brown.train -- bigram model (trained on brown.train) \n \t PERPLEXITY: {}".format(
    #     perplexity))
    #
    # # trigrams
    # trigram = ngram(SMALL_BROWN_TRAIN_TXT, 3)
    # perplexity = calculate_ngram_perplexity(SMALL_BROWN_DEV_TXT, trigram.vocabulary_space, trigram.probabilities, 3)
    # print("Perplexity Score for brown.dev -- trigram model (trained on brown.train) \n \t PERPLEXITY: {}".format(
    #     perplexity))
    # perplexity = calculate_ngram_perplexity(SMALL_BROWN_TRAIN_TXT, trigram.vocabulary_space, trigram.probabilities, 3)
    # print("Perplexity Score for brown.train -- trigram model (trained on brown.train) \n \t PERPLEXITY: {}".format(
    #     perplexity))
    #
    # # # trigrams w/ K-Smoothing
    # k = 10
    # n = 3
    # trigram_smooth = ngram(SMALL_BROWN_TRAIN_TXT, n, k)
    # perplexityDev = calculate_ngram_perplexity(SMALL_BROWN_DEV_TXT, trigram_smooth.vocabulary_space,
    #                                            trigram_smooth.probabilities, n, smoothed=True)
    # perplexityTrain = calculate_ngram_perplexity(SMALL_BROWN_TRAIN_TXT, trigram_smooth.vocabulary_space,
    #                                              trigram_smooth.probabilities, n, smoothed=True)
    # print(
    #     "Trigram model trained on brown.train\n"
    #     "Perplexity Score for \t brown.dev \t\t\t    brown.train\n "
    #     "\t\tPERPLEXITY: {}   |\t {} \t k={}".format(perplexityDev, perplexityTrain, k))
    #
    # k = 1
    # trigram_smooth = ngram(SMALL_BROWN_TRAIN_TXT, n, k)
    # perplexityDev = calculate_ngram_perplexity(SMALL_BROWN_DEV_TXT, trigram_smooth.vocabulary_space,
    #                                            trigram_smooth.probabilities, n, smoothed=True)
    # perplexityTrain = calculate_ngram_perplexity(SMALL_BROWN_TRAIN_TXT, trigram_smooth.vocabulary_space,
    #                                              trigram_smooth.probabilities, n, smoothed=True)
    # print("\t\tPERPLEXITY: {}   |\t {} \t k={}".format(perplexityDev, perplexityTrain, k))
    #
    # k = .1
    # trigram_smooth = ngram(SMALL_BROWN_TRAIN_TXT, n, k)
    # perplexityDev = calculate_ngram_perplexity(SMALL_BROWN_DEV_TXT, trigram_smooth.vocabulary_space,
    #                                            trigram_smooth.probabilities, n, smoothed=True)
    # perplexityTrain = calculate_ngram_perplexity(SMALL_BROWN_TRAIN_TXT, trigram_smooth.vocabulary_space,
    #                                              trigram_smooth.probabilities, n, smoothed=True)
    # print("\t\tPERPLEXITY: {}   |\t {} \t k={}".format(perplexityDev, perplexityTrain, k))
    # k = .01
    # trigram_smooth = ngram(SMALL_BROWN_TRAIN_TXT, n, k)
    # perplexityDev = calculate_ngram_perplexity(SMALL_BROWN_DEV_TXT, trigram_smooth.vocabulary_space,
    #                                            trigram_smooth.probabilities, n, smoothed=True)
    # perplexityTrain = calculate_ngram_perplexity(SMALL_BROWN_TRAIN_TXT, trigram_smooth.vocabulary_space,
    #                                              trigram_smooth.probabilities, n, smoothed=True)
    # print("\t\tPERPLEXITY: {}   |\t {} \t k={}".format(perplexityDev, perplexityTrain, k))
    # k = .001
    # trigram_smooth = ngram(SMALL_BROWN_TRAIN_TXT, n, k)
    # perplexityDev = calculate_ngram_perplexity(SMALL_BROWN_DEV_TXT, trigram_smooth.vocabulary_space,
    #                                            trigram_smooth.probabilities, n, smoothed=True)
    # perplexityTrain = calculate_ngram_perplexity(SMALL_BROWN_TRAIN_TXT, trigram_smooth.vocabulary_space,
    #                                              trigram_smooth.probabilities, n, smoothed=True)
    # print("\t\tPERPLEXITY: {}   |\t {} \t k={}".format(perplexityDev, perplexityTrain, k))
    # k = .0001
    # trigram_smooth = ngram(SMALL_BROWN_TRAIN_TXT, n, k)
    # perplexityDev = calculate_ngram_perplexity(SMALL_BROWN_DEV_TXT, trigram_smooth.vocabulary_space,
    #                                            trigram_smooth.probabilities, n, smoothed=True)
    # perplexityTrain = calculate_ngram_perplexity(SMALL_BROWN_TRAIN_TXT, trigram_smooth.vocabulary_space,
    #                                              trigram_smooth.probabilities, n, smoothed=True)
    # print("\t\tPERPLEXITY: {}   |\t {} \t k={}".format(perplexityDev, perplexityTrain, k))
    print("*** PART 4.2 ***")
    lamba1 = .4
    lamba2 = .4
    lamba3 = .2
    trigram_linear_model = linear_interpolation(SMALL_BROWN_TRAIN_TXT)
    perplexity = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, SMALL_BROWN_DEV_TXT)
    print(perplexity)

