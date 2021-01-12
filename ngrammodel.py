from config import pad_sentence, UNK_
from unigrammodel import UNK_THRESHOLD, MAX_UNKS


class ngram:
    def __init__(self, training_data_path, ngram_size):
        self.total_ngrams = 0
        self.ngram_key_occurrence: dict = {}
        self.ngram_sighting: dict = {}
        self.ngram_size = ngram_size
        self.extract_vocab(training_data_path, self.ngram_size)
        self.probabilities: dict = {}
        self.calculate_probabilities()

    def extract_vocab(self, training_data_path, ngram_size):
        training_data = open(training_data_path, "r")
        sentence = training_data.readline()
        while sentence:
            sentence_tokens = pad_sentence(sentence, ngram_size - 1).split()

            for i in range(len(sentence_tokens)):
                # test for if this is a valid ngram with this size
                if i < len(sentence_tokens) - (
                        ngram_size - 1):  # don't overstep into late sentence words who would include [STOP]
                    ngram_key = tuple(sentence_tokens[i:i+ngram_size-1])
                    ngram_tuple = tuple(sentence_tokens[i:i+ngram_size])

                    self.ngram_key_occurrence[ngram_key] = \
                        self.ngram_key_occurrence.get(ngram_key, 0) + 1
                    self.ngram_sighting[ngram_tuple] = self.ngram_sighting.get(ngram_tuple,
                                                                               0) + 1  # how many times have seen this ngram
                    self.total_ngrams += 1  # count up the total number of ngrams we see
            sentence = training_data.readline()
        training_data.close()

    def calculate_probabilities(self):
        #add an unk space
        unk_nom = 0
        unk_denom = 0
        for ngram_sight in self.ngram_sighting.keys():
            ngram_key = ngram_sight[0:len(ngram_sight)-1]

            if self.ngram_sighting[ngram_sight] <= UNK_THRESHOLD and unk_nom < MAX_UNKS:
                unk_nom += self.ngram_sighting[ngram_sight] # add all these things were unk'ing
                unk_denom += self.ngram_key_occurrence[ngram_key] # and add all the occurrences of the key
            else:
                self.probabilities[ngram_sight] = self.ngram_sighting[ngram_sight] / self.ngram_key_occurrence[ngram_key]
        if unk_denom > 0:
            # we found items below our unk threshold
            if self.ngram_size != 1:
                self.probabilities[(UNK_,)] = unk_nom / unk_denom
            else:
                self.probabilities[(UNK_,)] = unk_nom / self.total_ngrams


    def count_ngrams(self):
        return self.total_ngrams

    def distinct_ngrams(self):
        return len(self.ngram_sighting)
