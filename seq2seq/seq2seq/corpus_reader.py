# -*- coding: utf-8 -*-
from codecs import open

import sys
sys.path.append('..')
from reader import Reader, PAD_TOKEN, EOS_TOKEN, GO_TOKEN
from utils.io_utils import get_logger
logger = get_logger(__name__)

def save_word_dict(dict_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("%s\t%d\n" % (k, v))

def load_word_dict(save_path):
    dict_data = dict()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split()
            try:
                dict_data[items[0]] = int(items[1])
            except IndexError:
                logger.error('error', line)
    return dict_data

class CGEDReader(Reader):
    """
    Read CGED data set
    """
    UNKNOWN_TOKEN = 'UNK'

    def __init__(self, train_path=None, token_2_id=None):
        super(CGEDReader, self).__init__(
            train_path=train_path, token_2_id=token_2_id,
            special_tokens=[PAD_TOKEN, GO_TOKEN, EOS_TOKEN, CGEDReader.UNKNOWN_TOKEN])
        self.UNKNOWN_ID = self.token_2_id[CGEDReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            while True:
                line_src = f.readline()
                line_dst = f.readline()
                if not line_src or len(line_src) < 5:
                    break
                source = line_src.lower()[5:].strip().split()
                target = line_dst.lower()[5:].strip().split()
                yield source, target

    def unknown_token(self):
        return CGEDReader.UNKNOWN_TOKEN

    def read_tokens(self, path, is_infer=False):
        i = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # Input the correct text, which start with 0
                if i % 2 == 1:
                    if line and len(line) > 5:
                        yield line.lower()[5:].strip().split()
                i += 1

    @staticmethod
    def read_vocab(input_texts):
        vocab = {PAD_TOKEN, EOS_TOKEN, GO_TOKEN}
        for line in input_texts:
            for char in line:
                if char not in vocab:
                    vocab.add(char)
        return sorted(list(vocab))