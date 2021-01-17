from ngrammodel import ngram

"""
For now, this only works for trigrams but could be extended
"""
class linear_interpolation:
    def __init__(self, path_to_training_data):

        self.unigram = ngram(path_to_training_data, 1)
        self.bigram = ngram(path_to_training_data, 2)
        self.trigram = ngram(path_to_training_data, 3)

    def get_prob(self, trigram_tuple, lambda1, lambda2, lambda3):
        unigram = trigram_tuple[2:3]  # get Xi
        bigram = trigram_tuple[1:3]  # prob of Xi | Xi-1
        trigram = trigram_tuple  # prob of Xi | Xi-1, Xi-2
        unigram_prob_prime = self.unigram.probabilities[unigram] * lambda3  # key error is impossible (vocab bound)
        bigram_prob_prime = self.bigram.probabilities.get(bigram, 0) * lambda2  # Use GET for ngrams because 0 is valid
        trigram_prob_prime = self.trigram.probabilities.get(trigram, 0) * lambda1  # GET avoids key error
        p_prime = trigram_prob_prime + bigram_prob_prime + unigram_prob_prime # linear regression smoothed prob
        return p_prime

    def get_vocab(self):
        return self.unigram.vocabulary_space  # vocab is the same but use unigram since its more intuitive
