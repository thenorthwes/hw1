from unittest import TestCase

from config import STOP_, pad_sentence, START_


class Test(TestCase):
    def test_pad_sentence_adds_whitespace_and_special_stop(self):
        self.assertEqual("sentence " + STOP_,pad_sentence("sentence"))

    def test_pad_sentence_adds_starts_if_asked(self):
        self.assertEqual(START_ + " sentence " + STOP_, pad_sentence("sentence",1))
        self.assertEqual(START_ + " " + START_ + " sentence " + STOP_, pad_sentence("sentence",2))
        self.assertEqual(START_ + " " + START_ + " " + START_ + " sentence " + STOP_, pad_sentence("sentence",3))


