#Modified from @laurent-dinh
#https://github.com/laurent-dinh/dl_tutorials/blob/ef6d34d847ee4c631b0d28e72c152e171d4bc308/fuel/datasets/imdb.py
import numpy as np
import os
from collections import OrderedDict
import cPickle

import numpy
import theano

from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes
import glob

from fuel.transformers import Transformer
import pandas as pd
import csv

@do_not_pickle_attributes('indexables')
class IMDB(IndexableDataset):
    """
    IMDB dataset from the deeplearning tutorial.
    """

    provides_sources = ('features', 'targets')
    folder = 'imdb'
    filename_data = {'train': 'imdb_train.pkl',
            'test': 'imdb_test.pkl'}
    filename_dict = 'imdb.dict.pkl'
    num_examples = 0

    def __init__(self, which_set, **kwargs):
        self.which_set = which_set

        super(IMDB, self).__init__(
            OrderedDict(zip(self.provides_sources,
                            self._load_imdb())),
            **kwargs
        )

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provides_sources, self._load_imdb())
                           if source in self.sources]

    def _load_imdb(self):
        dir_path = os.path.join(config.data_path, self.folder)
        data_path = os.path.join(dir_path, self.filename_data[self.which_set])
        with open(data_path, 'r') as f:
            features, targets = cPickle.load(f)
        with open(os.path.join(dir_path, self.filename_dict), 'r') as f:
            self.dict = cPickle.load(f)

        features = numpy.array([numpy.array(s, dtype='int32')
                                for s in features])
        targets = numpy.array(targets, dtype='int32').reshape((-1, 1))

        self.inv_dict = {v: k for k, v in self.dict.items()}

        self.num_examples = features.shape[0]

        return (features, targets)

@do_not_pickle_attributes('indexables')
class IMDBText(IndexableDataset):
    """
    IMDB dataset from the deeplearning tutorial.
    """

    provides_sources = ('features', 'targets')
    folder = 'aclImdb'
    num_examples = 0

    def __init__(self, which_set, **kwargs):
        self.which_set = which_set

        super(IMDBText, self).__init__(
            OrderedDict(zip(self.provides_sources,
                            self._load_imdb())),
            **kwargs
        )

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provides_sources, self._load_imdb())
                           if source in self.sources]

    def _load_imdb(self):
        dir_path = os.path.join(config.data_path, self.folder)
        data_path = os.path.join(dir_path, self.which_set)

        pos_path = os.path.join(data_path, 'pos')
        neg_path = os.path.join(data_path, 'neg')

        files = glob.glob(pos_path+'/*.txt')
        pos_strings = [open(f, 'r').read() for f in files]
        pos_labels = np.ones(len(files))

        files = glob.glob(neg_path+'/*.txt')
        neg_strings = [open(f, 'r').read() for f in files]
        neg_labels = np.zeros(len(files))

        targets = np.hstack((pos_labels, neg_labels))
        targets = numpy.array(targets, dtype='int32').reshape((-1, 1))
        features = np.array(pos_strings + neg_strings)

        #n = 25000 / 2
        #features = features[n-1000:n+1000]
        #targets = targets[n-1000:n+1000]

        self.num_examples = len(features)

        return (features, targets)

from passage.preprocessing import tokenize

class GloveTransformer(Transformer):
    glove_folder = "glove"
    vector_dim = 0

    def __init__(self, glove_file, data_stream):
        super(GloveTransformer, self).__init__(data_stream)
        dir_path = os.path.join(config.data_path, self.glove_folder)
        data_path = os.path.join(dir_path, glove_file)
        raw = pd.read_csv(data_path, header=None, sep=' ', quoting=csv.QUOTE_NONE)
        #raw = pd.read_csv(data_path, nrows=400, header=None, sep=' ', quoting=csv.QUOTE_NONE)
        keys = raw[0].values
        vectors = raw[range(1, len(raw.columns))].values.astype(theano.config.floatX)
        self.vector_dim = vectors.shape[1]
        self.lookup = dict(zip(keys, vectors))

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        strings, target = next(self.child_epoch_iterator)
        strings = np.vectorize(str.lower)(strings)

        def process_string(s):
            tokens = tokenize(s)

            output = np.zeros((len(tokens), self.vector_dim), dtype=theano.config.floatX)
            for i,t in enumerate(tokens):
                if t in self.lookup:
                    output[i, :] = self.lookup[t]
            return output

        outputs = [process_string(s) for s in strings]
        return outputs, target

#class TextTrimmer(Transformer):
    #def __init__(self, max_length=100, data_stream):
        #super(TextTrimmer, self).__init__(data_stream)
        #self.max_length = max_length

    #def get_data(self, request=None):
        #if request is not None:
            #raise ValueError

    #strings, target = next(self.child_epoch_iterator)

    #def process_string(s):
        #output = np.zeros((len(tokens), self.vector_dim), dtype=theano.config.floatX)
        #for i,t in enumerate(tokens):
            #if t in self.lookup:
                #output[i, :] = self.lookup[t]
        #return output

    #outputs = [process_string(s) for s in strings]


if __name__=="__main__":
    dataset = IMDBText("train")

    stream = dataset.get_example_stream()
    glove_version = "vectors.6B.50d.txt"
    transformer = GloveTransformer(glove_version, data_stream=stream)
    print transformer.get_epoch_iterator().next()



