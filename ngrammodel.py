from config import pad_sentence, UNK_, WORKSPACE_
from unigrammodel import UNK_THRESHOLD, MAX_UNKS

TRAINING_UNKED_DATA_ = WORKSPACE_ + "ngram-training_unked_data"

"""
Replaces all tokens (whitespace sep) with UNK unless contained in Dict Vocab
"""


def write_new_training_data(training_data_path, vocab: dict, out_path):
    # input file
    fin = open(training_data_path, "rt")
    # output file to write the result to
    fout = open(out_path, "wt")
    # for each line in the input file
    for line in fin:
        tokens = line.split()
        unkedLine = ""
        for token in tokens:
            if token in vocab:
                unkedLine += token + " "
            else:
                unkedLine += UNK_ + " "
        # read replace the string and write to output file
        fout.write(unkedLine + "\n")
    # close input and output files
    fin.close()
    fout.close()
    return TRAINING_UNKED_DATA_


class ngram:
    def __init__(self, training_data_path, ngram_size):
        self.total_ngrams = 0
        self.ngram_key_occurrence: dict = {}
        self.ngram_sighting: dict = {}
        self.ngram_size = ngram_size
        self.vocabulary_space: dict = {}  # used for figuring out all vocabs
        self.unk_map: dict = {}  # will be filled with no more than MAX UNKs based on low occurrence words in vocab space
        self.calculate_unkspace_and_vocab(training_data_path)
        path_to_unked_data = write_new_training_data(training_data_path, self.vocabulary_space, TRAINING_UNKED_DATA_)
        #  TODO Fix the path i pass in here -- i need some new file which has UNKS jammed in
        self.extract_vocab(path_to_unked_data, self.ngram_size)
        self.probabilities: dict = {}
        self.calculate_probabilities()

    def extract_vocab(self, training_data_path, ngram_size):
        training_data = open(training_data_path, "r")
        sentence = training_data.readline()
        # loop through first and unk low occurrence words
        while sentence:
            sentence_tokens = pad_sentence(sentence, ngram_size - 1).split()
            for i in range(len(sentence_tokens)):
                # test for if this is a valid ngram with this size
                if i < len(sentence_tokens) - (
                        ngram_size - 1):  # don't overstep into late sentence words who would include [STOP]
                    ngram_key = tuple(sentence_tokens[i:i + ngram_size - 1])
                    ngram_tuple = tuple(sentence_tokens[i:i + ngram_size])

                    self.ngram_key_occurrence[ngram_key] = \
                        self.ngram_key_occurrence.get(ngram_key, 0) + 1
                    self.ngram_sighting[ngram_tuple] = self.ngram_sighting.get(ngram_tuple,
                                                                               0) + 1  # how many times have seen this ngram
                    self.total_ngrams += 1  # count up the total number of ngrams we see
            sentence = training_data.readline()
        training_data.close()

    def calculate_probabilities(self):
        # add an unk space
        unk_nom = 0
        unk_denom = 0
        for ngram_sight in self.ngram_sighting.keys():
            ngram_key = ngram_sight[0:len(ngram_sight) - 1]

            if self.ngram_sighting[ngram_sight] <= UNK_THRESHOLD and unk_nom < MAX_UNKS and False:
                unk_nom += self.ngram_sighting[ngram_sight]  # add all these things were unk'ing
                unk_denom += self.ngram_key_occurrence[ngram_key]  # and add all the occurrences of the key
            else:
                self.probabilities[ngram_sight] = self.ngram_sighting[ngram_sight] / self.ngram_key_occurrence[
                    ngram_key]

    def count_ngrams(self):
        return self.total_ngrams

    def distinct_ngrams(self):
        return len(self.ngram_sighting)

    def calculate_unkspace_and_vocab(self, training_data_path):
        training_data = open(training_data_path, "r")
        sentence = training_data.readline()
        # loop through first and unk low occurrence words
        while sentence:
            sentence_tokens = sentence.split()
            for vocab_instance in sentence_tokens:
                # test for if this is a valid ngram with this size
                self.vocabulary_space[vocab_instance] = self.vocabulary_space.get(vocab_instance,
                                                                                  0) + 1  # count every word
                self.total_ngrams += 1  # count up the total number of ngrams we see
            sentence = training_data.readline()
        training_data.close()
        # make our unks
        self.vocabulary_space[UNK_] = 0
        for vocab_word in list(self.vocabulary_space.keys()):
            if self.vocabulary_space[vocab_word] <= UNK_THRESHOLD and self.vocabulary_space[UNK_] < MAX_UNKS:
                self.unk_map[vocab_word] = UNK_
                self.vocabulary_space.pop(vocab_word)
                self.vocabulary_space[UNK_] += 1  # count every word

# TODO the reason this works is because with START START UNK for my worst case -- in a new setence
## ACTUALLY I THNK ITS OK CAUSE WE PRE UNK THE UNSEEN TEXT
# if i see START START CAPITOL and i search my probs -- dont find start start capitol -- i know i must unk capitol and add capitol to my new unked words and
# then as i grab new tuples from the dev set i must UNK words from my dev + new unked words which would result in START UNK foo or UNK foo UNK
