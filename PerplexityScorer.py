# Helpers for scoring perplexity
"""
Calculate Perplexity returns a score for a language model
-- it takes an evaluation path evalPath and probabilityDistribution (assuming well formed w/ [UNK] defined
For a method of scoring the evalPath -- it is treated as one long sentence
"""
from math import log2, inf

from config import UNK_, pad_sentence, WORKSPACE_
from ngrammodel import write_new_training_data, get_unk_tuple

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
    return round(perplexity, 4)


def calculate_ngram_perplexity(eval_path: str, vocab: dict, probs: dict, ngram_size: int, smoothed=False,
                               report_mode=False) -> int:
    exponent = 0
    path_to_unked_data = write_new_training_data(eval_path, vocab, EVAL_UNKED_DATA_)
    eval_stream = open(path_to_unked_data, "r")
    sentence = eval_stream.readline()
    ngram_counter = 0
    corpus_size = 0
    while sentence:
        sentence_tokens = pad_sentence(sentence, ngram_size - 1).split()
        corpus_size += len(sentence.split()) + 1  # don't count start -- but do count stop
        for i in range(len(sentence_tokens)):
            ngram_key = tuple(sentence_tokens[i:i + ngram_size])  # slice to get ngram size tuples
            if probs.get(ngram_key):
                exponent += log2(probs[ngram_key])
            else:
                if not smoothed:
                    exponent += -inf
                else:
                    # universal smooth -- words we knew but didnt see together
                    exponent += log2(probs[get_unk_tuple(ngram_size)])  # the ngram sized ungram we add

            ngram_counter += 1
        sentence = eval_stream.readline()

    #  Perplexity is equal to 2 to the power of the negative `l`
    perplexity = 2 ** -(exponent / corpus_size)

    if report_mode: print("Perplexity Score: {}".format(perplexity))

    eval_stream.close()
    return round(perplexity, 4)


def calculate_linear_perplexity(linear_model, lamda1, lambda2, lambda3, eval_path) -> int:
    exponent = 0
    path_to_unked_data = write_new_training_data(eval_path, linear_model.get_vocab(), EVAL_UNKED_DATA_)
    eval_stream = open(path_to_unked_data, "r")
    sentence = eval_stream.readline()
    ngram_counter = 0
    corpus_size = 0
    ngram_size = 3
    while sentence:
        sentence_tokens = pad_sentence(sentence, ngram_size - 1).split()
        corpus_size += len(sentence.split()) + 1  # don't count start -- but do count stop
        for i in range(len(sentence_tokens)):
            if i + ngram_size - 1 < len(sentence_tokens):
                ngram_key = tuple(sentence_tokens[i:i + ngram_size])
                exponent += log2(linear_model.get_prob(ngram_key, lamda1, lambda2, lambda3))

            ngram_counter += 1
        sentence = eval_stream.readline()

    #  Perplexity is equal to 2 to the power of the negative `l`
    perplexity = 2 ** -(exponent / corpus_size)
    eval_stream.close()
    return round(perplexity, 4)
