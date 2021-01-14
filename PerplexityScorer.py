# Helpers for scoring perplexity
"""
Calculate Perplexity returns a score for a language model
-- it takes an evaluation path evalPath and probabilityDistribution (assuming well formed w/ [UNK] defined
For a method of scoring the evalPath -- it is treated as one long sentence
"""
from math import log2, inf

from config import UNK_, STOP_, pad_sentence, WORKSPACE_
from ngrammodel import write_new_training_data
EVAL_UNKED_DATA_ = WORKSPACE_ + "ngram-eval_unked_data"

def calculate_perplexity(eval_path: str, probs: dict, report_mode=False) -> int:
    perplexity = -1
    exponent = 0
    eval_stream = open(eval_path, "r")
    sentence = eval_stream.readline()
    corpus_size = 0
    unigram_counter = 0
    while sentence:
        sentence = pad_sentence(sentence)
        for token in sentence.split():
            exponent += log2(probs.get(token, probs[UNK_]))
            corpus_size += 1
            unigram_counter += 1
        sentence = eval_stream.readline()


    #  Perplexity is equal to 2 to the power of the negative `l`
    perplexity = 2 ** -(exponent / corpus_size)

    if report_mode: print("Perplexity Score: {}".format(perplexity))

    eval_stream.close()
    return perplexity


def calculate_ngram_perplexity(eval_path: str, vocab: dict, probs: dict, ngram_size:int, smoothed=False, report_mode=False) -> int:
    perplexity = -1
    exponent = 0
    path_to_unked_data = write_new_training_data(eval_path, vocab, EVAL_UNKED_DATA_)
    eval_stream = open(path_to_unked_data, "r")
    sentence = eval_stream.readline()
    ngram_counter = 0
    corpus_size = 0
    while sentence:
        sentence_tokens = pad_sentence(sentence, ngram_size - 1).split()
        corpus_size += len(sentence_tokens)
        for i in range(len(sentence_tokens)):
            ngram_key = tuple(sentence_tokens[i:i + ngram_size])
            if probs.get(ngram_key):
                exponent += log2(probs[ngram_key])
            else:
                if not smoothed:
                    exponent += -inf

            ngram_counter += 1
        sentence = eval_stream.readline()

    #  Perplexity is equal to 2 to the power of the negative `l`
    perplexity = 2 ** -(exponent / corpus_size)

    if report_mode: print("Perplexity Score: {}".format(perplexity))

    eval_stream.close()
    return perplexity
