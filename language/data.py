import os
import shutil
import json
import argparse
import random
from io import open
import dill
import torch
import nltk
from nltk import word_tokenize
from torchtext.datasets import TranslationDataset
import spacy
from torchtext.data import Field, BucketIterator

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class NaivePsychCorpus():
    def __init__(self, path, test_percent):
        self.dictionary = Dictionary()
        self.train, self.valid, self.test = self.read_data(path, test_percent)
        self.name = "naive"
        self.datasets = {
                "train": self.train,
                "val":self.valid,
                "test":self.test}

    def splits(self):
        train, valid, test =\
                  self.tokenize(self.train), self.tokenize(self.valid),\
                                                      self.tokenize(self.test)
        return train, valid, test

    def read_data(self, path, test_percent):
        with open(path, 'r') as f:
            data = json.load(f)

        # train, valid_test = self.split_dict(data, test_percent*2)
        # valid, test = self.split_dict(valid_test, 0.5)
        train, valid, test = self.split_dict_label(data)
        return train, valid, test

        # test_valid_size = int((len(keys)*args.test_percent)*2)
        # keys[-int((len(keys)*args.test_percent)*2):]
        # test_idx = test_valid[len(test_valid)//2:]
        # valid_idx = test_valid[:len(test_valid)//2]
        # train_idx = keys[:-int((len(keys)*args.test_percent)*2)]

    def split_dict_label(self, d, shuffle=False):
        """
        Looks for "train/dev/test" label and splits accordingly
        """
        train = {}
        valid = {}
        test = {}
        for idkey, story in d.items():
            if story["partition"] == 'train':
                train[idkey] = story
            elif story["partition"] == 'dev':
                valid[idkey] = story
            elif story["partition"] == 'test':
                test[idkey] = story
        return train, valid, test

    def split_dict_percent(self, d, percent, shuffle=True):
        """
        Return two dictionaries with `percent` being size of smaller dict
        """
        keys = list(d.keys())
        if shuffle:
            random.shuffle(keys)
        n = int(len(keys)*percent)
        d1_keys = keys[:n]
        d2_keys = keys[-n:]
        d1 = {}
        d2 = {}
        for key, value in d.items():
            if key in d1_keys:
                d1[key] = value
            else:
                d2[key] = value
        return d1, d2

    def create_torchtext_files(self, path):
        """
        Saves train.src and train.trg files for train/test/valid
        To be used with torchtext TranslationDataset object
        """
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)

        for name, data in self.datasets.items():
            src, trg, src_meta, trg_meta = self.extract_lines(data)
            src_path = os.path.join(path, name + ".src")
            trg_path = os.path.join(path, name + ".trg")
            src_path_meta = os.path.join(path, name + ".src_meta")
            trg_path_meta = os.path.join(path, name + ".trg_meta")
            self.list_to_file(src, src_path)
            self.list_to_file(trg, trg_path)
            self.list_to_file(src_meta, src_path_meta)
            self.list_to_file(trg_meta, trg_path_meta)

    def list_to_file(self, ls, path):
        if os.path.exists(path):
            os.remove(path)
        with open(path, "w") as f:
            for line in ls:
                f.write(line+"\n")
        print(f"Saved data to file {path}")

    def extract_lines(self, d):
        """
        Extract all lines of text per story. Last line in speparate list.
        """
        source = [] # Just series of sentences
        source_w_meta = [] # Sentences with emotions
        target = []
        target_w_meta = []
        emo_tag = " @emo@ " # space on purpose
        start_line_tag = "@sol@ " # space on purpose

        # loop stories
        for idkey, story in d.items():
            context = []
            context_meta = []
            # loops lines in story
            for line, sentence in story["lines"].items():
                # Just text
                context.append(sentence["text"])

                # Text with meta
                emotions = self.get_line_emotions(sentence["characters"])
                context_meta.append(start_line_tag + sentence["text"]\
                                                        + emo_tag + emotions)
            # Last sentence as target
            target.append(context[-1])
            source.append(" ".join(context[:-1]))

            # Last sentence as target
            target_w_meta.append(context_meta[-1])
            source_w_meta.append(" ".join(context_meta[:-1]))
        return source, target, source_w_meta, target_w_meta

    def get_line_emotions(self, d):
        """
        Return comma separated emotions only for characters which appear
        Args:
            d: sentence dictionary
        """
        emotions = ""
        for char, text in d.items():
            # Check if character appears
            if text["app"] == True:
                for annotator, emo in text["emotion"].items():
                    emotions += ', ' + ', '.join(emo["text"])

        # Return without leading ', '
        return emotions[2:]

    def tokenize(self, d):
        """
        Tokenizes dictionary of text.
        Returns Torch LongTensor with all word ids
        """
        data = []
        # Add words to the dictionary
        tokens = 0
        # loop stories
        for idkey, story in d.items():
            # Indicate beginning of a story.
            sequence = ['<bos>']
            # loops lines in story
            for line, sentence in story["lines"].items():
                words = word_tokenize(sentence["text"])
                sequence.extend(words)
                sequence.append('<eol>') # end of line
            # Indicate end of story
            sequence.append('<eos>')
            tokens += len(sequence)
            data.append(sequence)
            for word in sequence:
                self.dictionary.add_word(word)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        for line in data:
            for word in line:
                ids[token] = self.dictionary.word2idx[word]
                token += 1

        return ids

class NaiveDataset(TranslationDataset):
    """
    Naive Psychology dataset implemented as a torchtext dataset. Inheriting
    from TranslationDataset since this is used for Seq2Seq like translation.
    """
    name = "naive"
    @classmethod
    def splits(cls, exts, fields, root='.data/stories/story_commonsense/torchtext',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of the Multi30k dataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.

        One way to use this class:
        """
        path = kwargs['path']
        del kwargs['path']

        return super(NaiveDataset, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)

# Load tokenizers
spacy_en = spacy.load('en')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_naive(args):
    """
    Convenience function to load pickle or dataset
    """
    # if os.path.isfile(args.prepared_data):
        # print("Found data pickle, loading from {}".format(args.prepared_data))
        # with open(args.prepared_data, 'rb') as p:
            # d = dill.load(p)
            # train_iterator = d["train_iterator"]
            # valid_iterator = d["valid_iterator"]
            # test_iterator  = d["test_iterator"]
            # src            = d["src"]
            # trg            = d["trg"]
        # return train_iterator, valid_iterator, test_iterator, src, trg

    # Create torchtext Fields for each language
    # and assign the correct tokenizer function
    src = Field(tokenize = tokenize_en,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True,
                include_lengths = True)

    trg = Field(tokenize = tokenize_en,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    train_data, valid_data, test_data = NaiveDataset.splits(\
                    exts = (args.src_ext, args.trg_ext), fields = (src, trg))

    # Build vocabularies
    src.build_vocab(train_data, min_freq = 2)
    trg.build_vocab(train_data, min_freq = 2)
    print(f"Source vocab size: {len(src.vocab)}")
    print(f"Target vocab size: {len(trg.vocab)}")

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
         batch_size = args.batch_size,
         sort_within_batch = True,
         sort_key = lambda x : len(x.src),
         device = args.device)

    # with open(args.prepared_data, 'wb') as p:
        # d = {}
        # d["train_iterator"] = train_iterator
        # d["valid_iterator"] = valid_iterator
        # d["test_iterator"] = test_iterator
        # d["src"] = src
        # d["trg"] = trg
        # dill.dump(d, p)
        # print("Saved prepared data for future fast load to: {}".format(\
                                                    # args.prepared_data))
    return train_iterator, valid_iterator, test_iterator, src, trg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Datasets')
    parser.add_argument('--create_naive', action='store_true',default=False,
                        help='Create the Naive dataset text files')

    parser.add_argument('--commonsense_location',
                        default="language/.data/stories/story_commonsense",
                        help='Source of the commonsense dataset')

    parser.add_argument('--commonsense_target',
                        default="language/.data/stories/story_commonsense/torchtext",
                        help='Where to save the naive torchtext data')

    args = parser.parse_args()
    if args.create_naive:
        data_path = args.commonsense_location + '/json_version/annotations.json'
        corpus = NaivePsychCorpus(data_path, 0.10)
        corpus.create_torchtext_files(args.commonsense_target)

