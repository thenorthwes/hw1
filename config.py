UNK_ = "[unk]"
STOP_ = "~stop~"
START_ = "[start]"
WORKSPACE_ = "./workspace/"


def pad_sentence(sentence: str, start_count=0):
    padded_sentence = sentence + " " + STOP_
    for i in range(start_count):
        padded_sentence = START_ + " " + padded_sentence
    return padded_sentence


UNK_THRESHOLD = 1
MAX_UNKS = 322

