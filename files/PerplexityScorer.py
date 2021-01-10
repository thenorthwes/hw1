# Helpers for scoring perplexity
"""
Calculate Perplexity returns a score for a language model
-- it takes an evaluation path evalPath and probabilityDistribution (assuming well formed w/ [UNK] defined
For a method of scoring the evalPath -- it is treated as one long sentence
"""
from math import log2

from unigrammodel import UNK_, STOP_


def calculate_perplexity(eval_path: str, probs: dict) -> int:
    perplexity = -1
    exponent = 0
    eval_stream = open(eval_path, "r")
    sentence = eval_stream.readline()
    corpus_size = 0
    while sentence:
        for token in sentence.split():
            exponent += log2(probs.get(token, probs[UNK_]))
            corpus_size += 1
            sentence = eval_stream.readline()
    # Add stop a single time
    corpus_size += 1
    exponent += log2(probs[STOP_])

    #  Perplexity is equal to 2 to the power of the negative `l`
    perplexity = 2 ** -(exponent / corpus_size)

    print("Unigram Perplexity Score: {}".format(perplexity))
    eval_stream.close()
    return perplexity
