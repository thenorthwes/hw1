# HW 1 NLP 517
# Wes Saunders -- All rights reserved etc

# Reads a sentence from our file and prints the test data
import time

from PerplexityScorer import calculate_perplexity, calculate_ngram_perplexity, calculate_linear_perplexity
from linear_interpolation import linear_interpolation
from ngrammodel import ngram
from unigrammodel import *

SMALL_BROWN_TRAIN_TXT = "./files/CSEP517-HW1-Data-Small/brown.train.txt"
SMALL_BROWN_DEV_TXT = "./files/CSEP517-HW1-Data-Small/brown.dev.txt"
SMALL_BROWN_TEST_TXT = "./files/CSEP517-HW1-Data-Small/brown.test.txt"

FULL_BROWN_TRAIN_TXT = "./files/CSEP517-HW1-Data-Full/brown.train.txt"
FULL_BROWN_DEV_TXT = "./files/CSEP517-HW1-Data-Full/brown.dev.txt"
FULL_BROWN_DEV_TEST = "./files/CSEP517-HW1-Data-Full/brown.test.txt"


def part_one(train, dev, test):
    # Unigram Model was my first attempt -- ngram was ultimately could handle UNIGRAM but keeping to 'show my work'
    print("{:>10} {:>20} {:>20} {:>20}".format("", "brown.dev", "brown.train", "brown.test"))
    unigramModel = UnigramModel(train)
    perplexity_dev = calculate_perplexity(dev, unigramModel.probabilities)
    perplexity_train = calculate_perplexity(train, unigramModel.probabilities)
    perplexity_test = calculate_perplexity(test, unigramModel.probabilities)
    print("{:>10} {:=20} {:=20} {:=20}".format("unigram", perplexity_dev, perplexity_train, perplexity_test))
    # bigrams
    bigram = ngram(train, 2)
    perplexity_dev = calculate_ngram_perplexity(dev, bigram.vocabulary_space, bigram.probabilities, 2)
    perplexity_train = calculate_ngram_perplexity(train, bigram.vocabulary_space, bigram.probabilities,
                                                  2)
    perplexity_test = calculate_ngram_perplexity(test, bigram.vocabulary_space, bigram.probabilities, 2)
    print("{:>10} {:=20} {:=20} {:=20}".format("bigram", perplexity_dev, perplexity_train, perplexity_test))
    # trigrams
    trigram = ngram(train, 3)
    perplexity_dev = calculate_ngram_perplexity(dev, bigram.vocabulary_space, bigram.probabilities, 3)
    perplexity_train = calculate_ngram_perplexity(train, bigram.vocabulary_space, bigram.probabilities,
                                                  3)
    perplexity_test = calculate_ngram_perplexity(test, bigram.vocabulary_space, bigram.probabilities, 3)
    print("{:>10} {:=20} {:=20} {:=20}".format("trigram", perplexity_dev, perplexity_train, perplexity_test))


def part_two(train, dev, test):
    # # trigrams w/ K-Smoothing
    n = 3  # For playing with higher n-grams only
    print("{:>10} {:>20} {:>20} {:>20}".format("Add K =", "brown.dev", "brown.train", "brown.test"))
    k = 10
    for i in range(6):
        trigram_smooth = ngram(train, n, k)
        perplexity_dev = calculate_ngram_perplexity(dev, trigram_smooth.vocabulary_space,
                                                    trigram_smooth.probabilities, n, smoothed=True)
        perplexity_train = calculate_ngram_perplexity(train, trigram_smooth.vocabulary_space,
                                                      trigram_smooth.probabilities, n, smoothed=True)
        perplexity_test = calculate_ngram_perplexity(test, trigram_smooth.vocabulary_space,
                                                     trigram_smooth.probabilities, n, smoothed=True)
        print("{:>10} {:=20} {:=20} {:=20}".format(k, perplexity_dev, perplexity_train, perplexity_test))
        k = k / 10  # order of magnitude step down


def part_four_one(full_train, full_dev, full_test):
    part_one(full_train, full_dev, full_test)
    print("Add K w/ full data set")
    part_two(full_train, full_dev, full_test)


def part_four_two(train, dev, test):
    print("{:>7} {:>7} {:>7} {:>20} {:>20} {:>20}".format("λ1", "λ2", "λ3", "brown.dev", "brown.train", "brown.test"))
    lamba1 = .8
    lamba2 = .1
    lamba3 = .1
    trigram_linear_model = linear_interpolation(train)
    perplexity_dev = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, dev)
    perplexity_train = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, train)
    perplexity_test = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, test)
    print("{:>7} {:>7} {:>7} {:>20} {:>20} {:>20}".format(lamba1, lamba2, lamba3, perplexity_dev, perplexity_train,
                                                          perplexity_test))
    lamba1 = .1
    lamba2 = .8
    lamba3 = .1
    trigram_linear_model = linear_interpolation(train)
    perplexity_dev = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, dev)
    perplexity_train = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, train)
    perplexity_test = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, test)
    print("{:>7} {:>7} {:>7} {:>20} {:>20} {:>20}".format(lamba1, lamba2, lamba3, perplexity_dev, perplexity_train,
                                                          perplexity_test))
    lamba1 = .1
    lamba2 = .1
    lamba3 = .8
    trigram_linear_model = linear_interpolation(train)
    perplexity_dev = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, dev)
    perplexity_train = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, train)
    perplexity_test = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, test)
    print("{:>7} {:>7} {:>7} {:>20} {:>20} {:>20}".format(lamba1, lamba2, lamba3, perplexity_dev, perplexity_train,
                                                          perplexity_test))
    lamba1 = .01
    lamba2 = .01
    lamba3 = .98
    trigram_linear_model = linear_interpolation(train)
    perplexity_dev = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, dev)
    perplexity_train = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, train)
    perplexity_test = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, test)
    print("{:>7} {:>7} {:>7} {:>20} {:>20} {:>20}".format(lamba1, lamba2, lamba3, perplexity_dev, perplexity_train,
                                                          perplexity_test))
    lamba1 = .1
    lamba2 = .2
    lamba3 = .7
    trigram_linear_model = linear_interpolation(train)
    perplexity_dev = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, dev)
    perplexity_train = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, train)
    perplexity_test = calculate_linear_perplexity(trigram_linear_model, lamba1, lamba2, lamba3, test)
    print("{:>7} {:>7} {:>7} {:>20} {:>20} {:>20}".format(lamba1, lamba2, lamba3, perplexity_dev, perplexity_train,
                                                          perplexity_test))


if __name__ == '__main__':
    print("Homework 1 -- program starting")
    print("\n\t\t*********** Part 1 -- Unigram, Bigram, Trigram ***********\n")
    part_one(SMALL_BROWN_TRAIN_TXT, SMALL_BROWN_DEV_TXT, SMALL_BROWN_TEST_TXT)
    print("\n\t\t*********** Part 2 -- Trigram with Add-K Smoothing ***********\n")
    part_two(SMALL_BROWN_TRAIN_TXT, SMALL_BROWN_DEV_TXT, SMALL_BROWN_TEST_TXT)
    print("\n\t\t*********** Part 4.1 Using Full Data Sets ***********\n")
    part_four_one(FULL_BROWN_TRAIN_TXT, FULL_BROWN_DEV_TXT, FULL_BROWN_DEV_TEST)
    print("\n\t\t*********** Part 4.2 Linear Interpolation λ ***********\n")
    part_four_two(SMALL_BROWN_TRAIN_TXT, SMALL_BROWN_DEV_TXT, SMALL_BROWN_TEST_TXT)
    print("Homework 1 -- program complete. Run time: {} seconds".format((round(time.process_time(),2))))
