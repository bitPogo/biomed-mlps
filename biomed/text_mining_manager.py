import tensorflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras

from biomed.mlps_manager import MLPsManager


class TextMiningManager:
    def __init__(self, properties_manager):
        self.properties_manager = properties_manager
        self.batch_size = None
        self.nb_classes = None
        self.Y_train = None
        self.Y_test = None
        self.X_train = None
        self.X_test = None
        self.training_features = None
        self.test_features = None
        self.input_dim = None
        self.training_data = None
        self.test_data = None
        self.doid_unique = None
        self.mlpsm = MLPsManager(self.properties_manager)

    def _data_train_test_split(self, data):
        test_size = self.properties_manager.test_split_size
        training_data, test_data = train_test_split(data, test_size=test_size)
        return training_data, test_data

    def _tfidf_transformation(self, training_data, test_data):
        properties = self.properties_manager.tfidf_transformation_properties
        vectorizer = TfidfVectorizer(min_df=properties['min_df'], max_df=properties['max_df'],
                                     max_features=properties['max_features'], ngram_range=properties['ngram_range'],
                                     sublinear_tf=properties['sublinear_tf'])

        vectorizer = vectorizer.fit(training_data['text'])
        training_features = vectorizer.transform(training_data['text'])

        test_features = vectorizer.transform(test_data['text'])
        return training_features, test_features

    def _prepare_input_data(self, data):
        self.training_data, self.test_data = self._data_train_test_split(data)
        print("test_data shape", self.test_data.shape)
        self.doid_unique = data['doid'].unique()
        self.doid_unique.sort()
        self.training_features, self.test_features = self._tfidf_transformation(self.training_data, self.test_data)
        self.X_train = self.training_features.toarray()
        self.X_test = self.test_features.toarray()
        print('X_train shape:', self.X_train.shape)
        print('X_test shape:', self.X_test.shape)

    def _prepare_target_data(self, test_data, training_data, target_dimension: str):
        y_train = np.array(training_data[target_dimension])
        y_test = np.array(test_data[target_dimension])
        if target_dimension == 'doid':
            y_train = self.map_doid_values_to_sequential(y_train)
            y_test = self.map_doid_values_to_sequential(y_test)
        self.Y_train = tensorflow.keras.utils.to_categorical(y_train, len(self.doid_unique) + 1)
        self.Y_test = tensorflow.keras.utils.to_categorical(y_test, len(self.doid_unique) + 1)

    def _normalize_input_data(self):
        scale = np.max(self.X_train)
        self.X_train /= scale
        self.X_test /= scale
        mean = np.mean(self.X_train)
        self.X_train -= mean
        self.X_test -= mean

    def setup_for_input_data(self, data):
        self._prepare_input_data(data)
        self._normalize_input_data()
        self.input_dim = self.X_train.shape[1]

    def setup_for_target_dimension(self, target_dimension):
        if target_dimension == 'doid':
            self.nb_classes = len(self.doid_unique)
        else:
            self.nb_classes = 2
        self._prepare_target_data(self.test_data, self.training_data, target_dimension)

    def get_binary_mlp_predictions(self):
        self.mlpsm.build_binary_mlp(input_dim=self.input_dim, nb_classes=self.nb_classes)
        predictions = self.mlpsm.train_and_run_binary_mlp(X_train=self.X_train, X_test=self.X_test,
                                                          Y_train=self.Y_train)
        return predictions

    def map_doid_values_to_sequential(self, y_data):
        output = []
        for input_doid in y_data:
            for target_doid, input_doid_mapping in enumerate(self.doid_unique):
                if input_doid == input_doid_mapping:
                    output.append(target_doid)
        return output

    def map_doid_values_to_nonsequential(self, y_data):
        output = []
        for seq_doid in y_data:
            output.append(self.doid_unique[seq_doid])
        return output
