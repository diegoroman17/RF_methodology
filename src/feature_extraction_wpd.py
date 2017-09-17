import numpy as np
import pywt
from multiprocessing import Pool
import re
import scipy.io as sio
import os
from os.path import join
import pickle


def signal2wp_energy(signal, wavelet, max_level):
    wp_tree = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=max_level)
    level = wp_tree.get_level(max_level, order='freq')
    energy_coef = np.zeros((len(level),))

    for i, node in enumerate(level):
        energy_coef[i] = np.sqrt(np.sum(node.data ** 2)) / node.data.shape[0]

    return energy_coef


class Data_set:
    def __init__(self, path,
                 sensor_number=0,
                 name_acq='Measures',
                 wavelet_list=['db7', 'sym3', 'coif4', 'bior6.8', 'rbio6.8'],
                 max_level=6):
        self.name_acq = name_acq
        self.sensor_number = sensor_number
        self.wavelet_list = wavelet_list
        self.max_level = max_level
        self.features, self.labels = self.processing_dataset(path)

    def processing_dataset(self, path):
        files = []
        labels = []
        for dir in os.listdir(path):
            folder = join(path, dir)
            list_files = [join(folder, file) for file in os.listdir(folder)]
            labels.extend(os.listdir(folder))
            files.extend(list_files)

        features = []
        with Pool() as p:
            features.append(p.map(self.feature_extraction_file, files))
        labels = [re.split('R|F|L|P|.mat', label)[1:-1] for label in labels]
        return np.squeeze(np.array(features)), np.array(labels)

    def feature_extraction_file(self, file):
        print(file)
        data = sio.loadmat(file)
        signal = data['data'][self.name_acq][0][0][self.sensor_number]
        features = np.zeros((len(self.wavelet_list) * 2 ** self.max_level,))

        for i, wavelet in enumerate(self.wavelet_list):
            features[i * 2 ** self.max_level:(i + 1) * 2 ** self.max_level] = signal2wp_energy(signal,
                                                                                               wavelet,
                                                                                               self.max_level)
        return features


if __name__ == "__main__":
    from feature_extraction_wpd import Data_set

    dataset = Data_set('/home/dcabrera/Dropbox/mechanical_datasets/DB_001V0/Raw_Data_DB_001V0')
    with open('../data/features001V0.pickle', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
