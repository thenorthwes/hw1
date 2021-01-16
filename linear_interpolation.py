from ngrammodel import ngram


class linear_interpolation:
    def __init__(self, path_to_training_data):

        self.unigram = ngram(path_to_training_data, 1)
        self.bigram = ngram(path_to_training_data, 2)
        self.trigram = ngram(path_to_training_data, 3)

    def get_prob(self, trigram_tuple, lambda1, lambda2, lambda3):
        unigram = trigram_tuple[2:3]
        bigram = trigram_tuple[1:3]
        trigram = trigram_tuple
        unigram_prob_prime = self.unigram.probabilities[unigram] * lambda3  # accept key error since impossible (vocab bound)
        bigram_prob_prime = self.bigram.probabilities.get(bigram, 0) * lambda2  # Use GET for bigram / trigram because 0 is valid
        trigram_prob_prime = self.trigram.probabilities.get(trigram,0) * lambda1
        p_prime = trigram_prob_prime + bigram_prob_prime + unigram_prob_prime
        return p_prime

    def get_vocab(self):
        return self.unigram.vocabulary_space
