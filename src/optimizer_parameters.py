from read_dataset import read_datasets
from sklearn.ensemble import RandomForestClassifier
import copy
import numpy as np
from multiprocessing import Pool
from itertools import combinations
import pickle


class OptimizerRandomForest:
    def __init__(self, path, min_estimators=15, max_estimators=1000, random_state=123):
        self.random_state = random_state
        dataset, wavelet_list, max_level = read_datasets(path, random_state)
        self.dataset = dataset.train
        self.wavelet_list = wavelet_list
        self.max_level = max_level
        self.model = RandomForestClassifier(warm_start=True, oob_score=True, random_state=random_state)
        self.min_estimators = min_estimators
        self.max_estimators = max_estimators
        self.features_index = np.zeros((self.dataset.data.shape[1],), dtype=bool)
        self.oob_errors, self.subsets_families = self.search_best_wavelets()

    def search_best_wavelets(self):
        families = []
        for i in range(1, len(self.wavelet_list)):
            families.extend(combinations(range(len(self.wavelet_list)), i))
        families.extend([tuple(range(len(self.wavelet_list)))])

        oob_error = []
        for i in families:
            starts = np.array(i) * 2 ** self.max_level
            self.features_index[:] = False
            for start in starts:
                self.features_index[start:start + 2 ** self.max_level] = True
            oob_error.append(self.search_parameters())
        return oob_error, families

    def search_parameters(self):
        n_random_features = range(1, np.sum(self.features_index) + 1)
        with Pool() as p:
            oob_error = p.map(self.search_n_estimators, n_random_features)
        return np.array(oob_error)

    def search_n_estimators(self, n_random_features):
        print('number random features:', n_random_features)
        model = copy.deepcopy(self.model)
        model.set_params(max_features=n_random_features)
        oob_error = np.empty((self.max_estimators - self.min_estimators + 1,))
        oob_error[:] = np.NAN
        for i in range(self.min_estimators, self.max_estimators + 1):
            model.set_params(n_estimators=i)
            model.fit(self.dataset.data[:, self.features_index], self.dataset.labels[:, -1])
            oob_error[i - self.min_estimators] = 1 - model.oob_score_
        return oob_error


def main(path):
    from optimizer_parameters import OptimizerRandomForest
    optimizer = OptimizerRandomForest(path)
    with open('../data/DB_001V0_ooberror26062017.pickle', 'wb') as f:
        pickle.dump(optimizer, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main('../data/features001V0.pickle')
