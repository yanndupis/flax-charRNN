import collections

import numpy as np
import spacy

en = spacy.load("en")


def batchify(data, batch_size):
    n_batch = data.shape[0] // batch_size
    data = data[: n_batch * batch_size]
    data = data.reshape(batch_size, -1).transpose()
    return data


def get_batch(source, seq_len, i):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


def process_data(data, tokenizer, vocabulary, batch_size):
    tokenized_text = tokenizer.to_tokens(data, char_level=True)
    vocab = vocabulary.build_vocab(tokenized_text)
    numericalized = vocabulary.numericalize(tokenized_text)
    data_batchified = batchify(np.array(numericalized), batch_size)
    return data_batchified


class Vocab:
    def __init__(self):
        self.itos = None
        self.stoi = None

    def build_vocab(self, tokens):
        self.stoi = collections.defaultdict(
            None, {v: k for k, v in enumerate(set(tokens))}
        )
        self.itos = {self.stoi[k]: k for k in self.stoi}
        return self.stoi

    def numericalize(self, tokens):
        return [self.stoi[t] for t in tokens]

    def textify(self, nums):
        if isinstance(nums, int):
            return [self.itos[nums]]
        else:
            return [self.itos[n] for n in nums]


class Tokenizer:
    def to_tokens(self, text, char_level: True):
        if char_level:
            return list(text)
        else:
            return [w.text for w in en(text)]
