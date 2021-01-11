from unittest import TestCase
from ngrammodel import ngram

"""
Test data Shape:
    there are eight tokens
    on the two sentences
    
    Bigrams:                           
        START there                            
        there are                           
        are eight                           
        eight tokens                           
        tokens STOP                           
        START on                           
        on the                           
        the two                           
        two sentences                           
        sentences STOP                           
        
        trigrams:
        START START there 
        START there are
        there are eight
        are eight tokens
        eight tokens STOP
        START START on
        START on the
        on the two
        the two sentences
        two sentences STOP
"""
TESTDATA_TXT = "./files/test/testdata.txt"


class Testngram(TestCase):
    def test_Experiment_unigram(self):
        ngrammer = ngram(TESTDATA_TXT, 1)
        self.assertEqual(10, ngrammer.count_ngrams())
        self.assertEqual(9, ngrammer.distinct_ngrams())

    def test_counts_ngrams_correctly_for_bigram(self):
        ngrammer = ngram(TESTDATA_TXT, 2)
        self.assertEqual(10, ngrammer.count_ngrams())

    def test_counts_ngrams_correctly_for_trigram(self):
        ngrammer = ngram(TESTDATA_TXT, 3)
        self.assertEqual(10, ngrammer.count_ngrams())
