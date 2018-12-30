from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class Decorate:
    def __init__(self, n_estimators=100, max_iter=1000, r_factor=0.4):
        self.trees = []
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.r_factor = r_factor
        self.possible_labels = None

    def fit(self, features, labels):
        iter_num = 1
        ensemble_size = 1
        self.trees = []
        first_model = DecisionTreeClassifier(class_weight='balanced', max_depth=9)
        first_model.fit(features, labels)
        self.trees.append(first_model)
        self.possible_labels = labels.unique()
        current_error = self._compute_ensemble_error(features, labels)
        print("Current Error", current_error)
        while ensemble_size < self.n_estimators and iter_num < self.max_iter:
            print("Iteration number", iter_num)
            artificial_data = self._generate_artificial_data(features)
            artificial_labels = self._label_artificial_data(artificial_data)
            training_data = pd.concat([features, artificial_data], axis=0)
            training_labels = np.concatenate((labels.values, artificial_labels), axis=0)
            new_model = DecisionTreeClassifier(class_weight='balanced', max_depth=9)
            new_model.fit(training_data, training_labels)
            self.trees.append(new_model)
            error = self._compute_ensemble_error(features, labels)
            print("with new error", error)
            if error < current_error:
                current_error = error
                ensemble_size += 1
                print("Appending classifier ", ensemble_size)
            else:
                self.trees.remove(new_model)
            iter_num += 1

    def predict(self, samples):
        results = np.zeros(samples.shape[0])
        probas = self.predict_proba(samples)
        for sample_i in samples:
            results[sample_i] = probas[sample_i].index(max(probas[sample_i]))
        return results

    def predict_proba(self, samples):
        ensemble_size = len(self.trees)
        num_of_labels = len(self.possible_labels)
        probas = np.zeros((samples.shape[0], num_of_labels))
        for tree in self.trees:
            probas += tree.predict_proba(samples) / ensemble_size

        return probas

    def _compute_ensemble_error(self, samples, labels):
        probas = np.zeros((len(samples), len(labels.unique())))
        ensemble_size = len(self.trees)
        for tree in self.trees:
            probas += tree.predict_proba(samples) / ensemble_size

        errors = 0
        for proba_i in range(len(probas)):
            predicted = np.argmax(probas[proba_i])
            if predicted != labels[proba_i]:
                errors += 1

        return errors / len(samples)

    def _generate_artificial_data(self, features : pd.DataFrame):
        number_of_samples = int(features.shape[0] * self.r_factor)
        artificial_data = np.empty((number_of_samples, features.shape[1]))
        for col_index in range(len(features.columns)):
            column = features[features.columns[col_index]]
            if column.dtype == np.number:
                artificial_data[:, col_index] = np.random.normal(np.mean(column),
                                                                 np.std(column),
                                                                 size=number_of_samples)
            elif column.dtype == np.object_:
                unq_classes = column.nunique()
                unq_classes /= column.shape[0]
                artificial_data[:, col_index] = np.random.multinomial(features.shape[0],
                                                                      unq_classes,
                                                                      size=number_of_samples)
        artificial_data = pd.DataFrame(artificial_data, columns=features.columns)
        return artificial_data

    def _label_artificial_data(self, artificial_data):
        labels = np.zeros(artificial_data.shape[0])
        probabilities = self.predict_proba(artificial_data)

        for sample in range(len(probabilities)):
            normalized_probas = self._normalize_probas(probabilities[sample])
            for prob_index in range(len(normalized_probas)):
                normalized_probas[prob_index] = 1 / normalized_probas[prob_index]

                normalized_probas = normalized_probas / np.sum(normalized_probas)
            labels[sample] = np.random.choice(self.possible_labels, p=normalized_probas)
        return labels


    def _normalize_probas(self, probas):
        for proba in probas:
            if proba == 0:
                proba = 0.001

        sum_probas = np.sum(probas)
        return probas / sum_probas

