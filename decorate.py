from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class Decorate:
    def __init__(self, n_estimators=100, max_iter=1000, r_factor=0.4):
        self.trees = []
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.r_factor = r_factor

    def fit(self, features, labels):
        iter_num = 1
        ensemble_size = 1
        first_model = DecisionTreeClassifier()
        first_model.fit(features, labels)
        self.trees.append(first_model)
        current_error = self._compute_ensemble_error(features, labels)
        while ensemble_size < self.n_estimators and iter_num < self.max_iter:
            artificial_data = self._generate_artificial_data(features)
            artificial_labels = self._label_artificial_data(artificial_data)
            training_data = pd.concat([features, artificial_data], axis=1)
            training_labels = pd.concat([labels, artificial_labels], axis=1)

            new_model = DecisionTreeClassifier()
            new_model.fit(training_data, training_labels)
            self.trees.append(new_model)
            error = self._compute_ensemble_error(features, labels)
            if error < current_error:
                current_error = error
                ensemble_size += 1
            else:
                self.trees.remove(new_model)
            iter_num += 1

    def predict(self, samples):
        results = np.zeros(len(samples))
        ensemble_size = len(self.trees)
        num_of_labels = len(self.trees[0].predict_proba(samples[0]))
        for sample_i in range(len(samples)):
            probas = np.zeros(num_of_labels)
            for tree in self.trees:
                probas += tree.predict_proba(samples[sample_i]) / ensemble_size
            results[sample_i] = probas.index(max(probas))
        return results

    def _compute_ensemble_error(self, features, labels):
        probas = np.zeros((len(features), len(labels.unique())))
        ensemble_size = len(self.trees)
        for feature_i in range(len(features)):
            for tree in self.trees:
                probas[feature_i] += tree.predict_proba(features[feature_i]) / ensemble_size

        errors = 0
        for proba_i in range(len(probas)):
            predicted = probas[proba_i].index(max(probas[proba_i]))
            if predicted != labels[proba_i]:
                errors += 1

        return errors / len(features)

    def _generate_artificial_data(self, features : pd.DataFrame):
        artificial_data = np.empty(features.shape)
        number_of_samples = int(features.shape[0] * self.r_factor)
        for col_index in range(len(features.columns)):
            column = features[features.columns[col_index]]
            if column.dtype == np.numeric:
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
        #TODO: Label the artificial data correctly
        return artificial_data


