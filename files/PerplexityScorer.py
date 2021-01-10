# Helpers for scoring perplexity
"""
Calculate Perplexity returns a score for a language model
-- it takes an evaluation path evalPath and probabilityDistribution (assuming well formed w/ [UNK] defined
For a method of scoring the evalPath -- it is treated as one long sentence
"""
from math import log2

from unigrammodel import UNK_, STOP_


def calculate_perplexity(evalPath: str, probs: dict) -> int:
    perplexity = -1
    exponent = 0
    eval_stream = open(evalPath, "r")
    sentence = eval_stream.readline()
    corpusSize = 0
    while sentence:
        for token in sentence.split():
            exponent += log2(probs.get(token, probs[UNK_]))
            corpusSize += 1
            sentence = eval_stream.readline()
    # Add stop a single time
    corpusSize += 1
    exponent += log2(probs[STOP_])


    # Perplexity is equal to 2 to the power of the negative `l`
    perplexity = 2 ** -(exponent/corpusSize)

    # for validationword in validationGrams.getUnigrams().keys():
    #     perplexity

    print("Unigram Perplexity Score: {}".format(perplexity))
    return perplexity
