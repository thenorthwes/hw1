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
        self.unigramcounter: dict  # set up when ready
        self.probabilities: dict
        self.vocabsize: int = 0  # assume empty vocab size to start
        self.train(trainingDataPath)
        self.reportTraining()

    # main worker for unigram model training
    def train(self, trainingDataPath):

        # set up our model structure
        self.unigramcounter = {}  # use a dictionary for convenience

        trainingDataFile = open(trainingDataPath, "r")  # open a handle to our training corpus (read mode)
        sentence = trainingDataFile.readline()  # get the first sentence

        while sentence:
            sentence = pad_sentence(sentence)
            self.train_sentence(sentence)  # update our model with this sentence
            sentence = trainingDataFile.readline()  # go to the next line
        trainingDataFile.close()

    '''
    Train structures on a sentence -- takes a tokenized sentence (white space indicates a new token)
    and process into our unigram model
    '''

    def train_sentence(self, sentence):
        for token in sentence.split():
            # get a token and if we are first meeting it, default to 0
            self.unigramcounter[token] = self.unigramcounter.get(token,0) + 1
            self.vocabsize += 1

    def reportTraining(self):
        print("Number of unigrams: {}".format(len(self.unigramcounter.keys())))
        print("Vocabulary Size: {}", self.vocabsize)
        print("Number of occurrences of 'the': {}".format(self.unigramcounter["the"]))
        print("Number of occurrences of '.': {}".format(self.unigramcounter["."]))
        print("Number of occurrences of special stop: {}".format(self.unigramcounter[("%s" % STOP_)]))

    def getUnigrams(self):
        return self.unigramcounter

    def getVocabSize(self):
        return self.vocabsize

class UnigramModel:
    def __init__(self, trainingDataPath: str) -> None:
        self.probabilities: dict
        self.unigrams = Unigrams(trainingDataPath)
        self.learn()
        self.reportLearning()

    def learn(self):
        # probability is the number of encounters / total tokens
        # Get our Unigrams and Vocab
        unigrams = self.unigrams.getUnigrams()
        vocabsize = self.unigrams.getVocabSize()
        # add the UNK place holder -- start at 0 occurrences
        unigrams[UNK_] = 0

        self.probabilities = {}

        for unigram in unigrams.keys():
            self.probabilities[unigram] = unigrams[unigram] / vocabsize
            if unigrams[unigram] <= 1:  # [unk] the single occurrence unigrams
                unigrams[UNK_] = unigrams.get(UNK_, 0) + unigrams[unigram]
        self.probabilities[UNK_] = unigrams[UNK_] / vocabsize

    def reportLearning(self):
        print("Probability of 'the': {}".format(self.probabilities["the"]))
        print("Probability of [unk]: {}".format(self.probabilities[UNK_]))

