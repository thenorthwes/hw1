# Implements unigram language model
# primary uses are 1) train with a file (passed as a path) to train probabilities of words
# and 2) evaluate against another file -- to evaluate perplexity scores (fitness of model)
from math import log2

UNK_ = "[unk]"
STOP_ = "~stop~"


def pad_sentence(sentence: str):
    return sentence + " " + STOP_

class Unigrams:
    def __init__(self, trainingDataPath: str) -> None:
        self.unigram_counter: dict = None  # set up when ready
        self.probabilities: dict
        self.vocab_size: int = 0  # assume empty vocab size to start
        self.extract_vocab(trainingDataPath)
        self.report_training()

    # main worker for unigram model training
    def extract_vocab(self, training_data_path):

        # set up our model structure
        self.unigram_counter = {}  # use a dictionary for convenience

        training_data_file = open(training_data_path, "r")  # open a handle to our training corpus (read mode)
        sentence = training_data_file.readline()  # get the first sentence

        while sentence:
            sentence = pad_sentence(sentence)
            self.consume_sentence(sentence)  # update our model with this sentence
            sentence = training_data_file.readline()  # go to the next line
        training_data_file.close()

    '''
    Train structures on a sentence -- takes a tokenized sentence (white space indicates a new token)
    and process into our unigram model
    '''

    def consume_sentence(self, sentence):
        for token in sentence.split():
            # get a token and if we are first meeting it, default to 0
            self.unigram_counter[token] = self.unigram_counter.get(token, 0) + 1
            self.vocab_size += 1

    def report_training(self):
        print("Number of unigrams: {}".format(len(self.unigram_counter.keys())))
        print("Vocabulary Size: {}", self.vocab_size)
        print("Number of occurrences of 'the': {}".format(self.unigram_counter["the"]))
        print("Number of occurrences of '.': {}".format(self.unigram_counter["."]))
        print("Number of occurrences of special stop: {}".format(self.unigram_counter[("%s" % STOP_)]))

    def get_unigrams(self):
        return self.unigram_counter

    def get_vocab_size(self):
        return self.vocab_size

class UnigramModel:
    def __init__(self, trainingDataPath: str) -> None:
        self.probabilities: dict = None  # set up later
        self.unigrams = Unigrams(trainingDataPath)
        self.learn()
        self.report_learning()

    def learn(self):
        # probability is the number of encounters / total tokens
        # Get our Unigrams and Vocab
        unigrams = self.unigrams.get_unigrams()
        vocab_size = self.unigrams.get_vocab_size()
        # add the UNK place holder -- start at 0 occurrences
        unigrams[UNK_] = 0

        self.probabilities = {}

        for unigram in unigrams.keys():
            self.probabilities[unigram] = unigrams[unigram] / vocab_size
            if unigrams[unigram] <= 1:  # [unk] the single occurrence unigrams
                unigrams[UNK_] = unigrams.get(UNK_, 0) + unigrams[unigram]
        self.probabilities[UNK_] = unigrams[UNK_] / vocab_size

    def report_learning(self):
        print("Probability of 'the': {}".format(self.probabilities["the"]))
        print("Probability of [unk]: {}".format(self.probabilities[UNK_]))

