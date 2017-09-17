import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import copy


class DataSet(object):
    def __init__(self,
                 data,
                 labels,
                 dtype=dtypes.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype != dtypes.float32:
            raise TypeError('Invalid data dtype %r, expected float32' % dtype)

        assert data.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (data.shape, labels.shape))

        self._num_examples = data.shape[0]
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def data_by_condition(self, condition):
        return self._data[(self._labels == condition).all(axis=1)]

    def num_examples_by_condition(self, condition):
        return np.count_nonzero((self._labels == condition).all(axis=1))

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]

    def batch_test(self, condition):
        print(self._data.shape)
        return self._data[(self._labels == condition).all(axis=1)], \
               self._labels[(self._labels == condition).all(axis=1)]


def read_datasets(path_data, random_state, test_size=0.3):
    with open(path_data, 'rb') as f:
        dataset = pickle.load(f)

    data = dataset.features
    print('number of instances:', data.shape[0])
    print('number of features:', data.shape[1])
    labels = dataset.labels

    data_train, data_test, labels_train, labels_test = train_test_split(data,
                                                                        labels[:, 1:],
                                                                        test_size=test_size,
                                                                        stratify=labels[:, -1],
                                                                        random_state=random_state)
    data_valid = copy.copy(data_test)
    labels_valid = copy.copy(labels_test)
    train = DataSet(data_train, labels_train)
    validation = DataSet(data_valid, labels_valid)
    test = DataSet(data_test, labels_test)

    return base.Datasets(train=train, validation=validation, test=test), dataset.wavelet_list, dataset.max_level
