import pandas as pd
import unittest
import tensorflow as tf
import numpy as np
from copy import deepcopy
from typing import Sequence, Optional


class MetaData:
    CATEGORICAL_DATA_TYPE = 'categorical'
    NUMERICAL_DATA_TYPE = 'numerical'
    UNKNOWN_DATA_TYPE = 'unknown'

    def __init__(self, data: pd.DataFrame):
        self.uncategorized_columns = list(data)
        self.categorical_columns = list()
        self.numerical_columns = list()

    def __str__(self):
        stringy_self = ''

        for categorical_column in self.categorical_columns:
            stringy_self = stringy_self + '\n' + categorical_column + ': ' + MetaData.CATEGORICAL_DATA_TYPE

        for numerical_column in self.numerical_columns:
            stringy_self = stringy_self + '\n' + numerical_column + ': ' + MetaData.NUMERICAL_DATA_TYPE

        for unknown_column in self.uncategorized_columns:
            stringy_self = stringy_self + '\n' + unknown_column + ': ' + MetaData.UNKNOWN_DATA_TYPE

        return stringy_self

    def define_categorical_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.categorical_columns, column_name)

    def define_numerical_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.numerical_columns, column_name)

    def define_uncategorized_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.uncategorized_columns, column_name)

    def add_column_to_type(self, type_list: list, column_name: str):
        self.remove_column(column_name)
        if column_name not in type_list:
            type_list.append(column_name)

    def remove_column(self, column: str):
        if column in self.numerical_columns:
            self.numerical_columns.remove(column)

        if column in self.categorical_columns:
            self.categorical_columns.remove(column)

        if column in self.uncategorized_columns:
            self.uncategorized_columns.remove(column)

    def get_column_type(self, column: str) -> Optional[str]:
        if column in self.categorical_columns:
            return self.CATEGORICAL_DATA_TYPE

        if column in self.numerical_columns:
            return self.NUMERICAL_DATA_TYPE

        if column in self.uncategorized_columns:
            return self.UNKNOWN_DATA_TYPE

        return None


class TestMetaData(unittest.TestCase):

    def setUp(self):
        self._data = {
            'numerical_1': [1, 2],
            'numerical_2': [3, 4],
            'numerical_3': [3, 4],
            'categorical_1': [7, 8],
            'categorical_2': [5, 6],
            'categorical_3': [5, 6],
            'unknown_1': [9, 10]
        }

        self._dataframe = pd.DataFrame(data=self._data)
        self.meta_data = MetaData(self._dataframe)

    def test_categorize_columns(self):
        # categorize all columns
        self.meta_data.define_categorical_columns(['categorical_1', 'categorical_2'])
        self.meta_data.define_numerical_columns(['numerical_1', 'numerical_2'])
        self.meta_data.define_categorical_columns(['categorical_3'])
        self.meta_data.define_numerical_columns(['numerical_3'])

        # scramble the columns
        self.meta_data.define_categorical_columns(['numerical_1', 'numerical_2'])
        self.meta_data.define_numerical_columns(['categorical_1', 'categorical_2', 'unknown_1'])

        # categorize the columns again
        self.meta_data.define_categorical_columns(['categorical_1', 'categorical_2'])
        self.meta_data.define_numerical_columns(['numerical_1', 'numerical_2'])
        self.meta_data.define_categorical_columns(['categorical_3'])
        self.meta_data.define_numerical_columns(['numerical_3'])
        self.meta_data.define_uncategorized_columns(['unknown_1'])

        self.assertListEqual(['categorical_1', 'categorical_2', 'categorical_3'], self.meta_data.categorical_columns)
        self.assertListEqual(['numerical_1', 'numerical_2', 'numerical_3'], self.meta_data.numerical_columns)
        self.assertListEqual(['unknown_1'], self.meta_data.uncategorized_columns)

    def test_to_string(self):
        self.meta_data.define_categorical_columns(['categorical_1', 'categorical_2', 'categorical_3'])
        self.meta_data.define_numerical_columns(['numerical_1', 'numerical_2', 'numerical_3'])

        expected_string = '\ncategorical_1: ' + MetaData.CATEGORICAL_DATA_TYPE + '\ncategorical_2: ' + MetaData.CATEGORICAL_DATA_TYPE + '\ncategorical_3: ' + MetaData.CATEGORICAL_DATA_TYPE + '\nnumerical_1: ' + MetaData.NUMERICAL_DATA_TYPE + '\nnumerical_2: ' + MetaData.NUMERICAL_DATA_TYPE + '\nnumerical_3: ' + MetaData.NUMERICAL_DATA_TYPE + '\nunknown_1: ' + MetaData.UNKNOWN_DATA_TYPE
        stringified_metadata = str(self.meta_data)
        self.assertEqual(expected_string, stringified_metadata, 'Metadata __str__ function does not meet expectations.')


class DataModel:
    _dataframe: pd.DataFrame
    _tf_feature_columns: list

    def __init__(self, data: pd.DataFrame):
        self.metadata = MetaData(data)
        self._dataframe = data
        self.feature_columns_names = []
        self.target_column_name = None

    def get_dataframe(self):
        return self._dataframe

    def set_dataframe(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe

    def validate_columns(self, column_names: Sequence):
        for name in column_names:
            if name not in self._dataframe.columns:
                raise RuntimeError("'{}' not in dataset.".format(name))

    def get_all_column_categories(self, column: str):
        return self._dataframe[column].unique().tolist()

    def set_tf_feature_columns(self, feature_columns: list):
        column_names = []
        for column in feature_columns:
            column_names.append(column.name)

        self._tf_feature_columns = feature_columns
        self.feature_columns_names = column_names

    def get_tf_feature_columns(self):
        return self._tf_feature_columns

    def set_feature_columns(self, feature_column_names: list):
        self.validate_columns(feature_column_names)
        self.feature_columns_names = feature_column_names

    def get_feature_columns(self):
        self.validate_columns(self.feature_columns_names)

        return self._dataframe[self.feature_columns_names]

    def set_target_column(self, target_column_name: str):
        self.validate_columns([target_column_name])
        self.target_column_name = target_column_name

    def get_target_column(self):
        self.validate_columns([self.target_column_name])

        return self._dataframe[self.target_column_name]


class TestDataset(unittest.TestCase):

    def setUp(self):
        self._data = {'col1': [1, 2], 'col2': [3, 4], 'col4': [5, 6]}
        self._dataframe = pd.DataFrame(data=self._data)
        self._dataset = DataModel(self._dataframe)

    def test_validate_columns(self):
        with self.assertRaises(RuntimeError):
            self._dataset.validate_columns(['col3'])

    def test_feature_columns(self):
        intended_columns = ['col1', 'col2']
        self._dataset.set_feature_columns(intended_columns)

        feature_columns = self._dataset.get_feature_columns()
        result_columns = list(feature_columns.columns.values)

        self.assertEqual(result_columns, intended_columns)

    def test_target_column(self):
        intended_column = 'col1'
        self._dataset.set_target_column(intended_column)

        target_column = self._dataset.get_target_column()

        self.assertEqual(target_column.tolist(), self._data[intended_column])


class DataSetSplitter:
    _data_model: DataModel

    def __init__(self, data_model: DataModel):
        self._data_model = data_model

    def split_by_ratio(self, ratios: list):
        model_data = self._data_model.get_dataframe()

        ratios = np.array(ratios)
        data_length = len(self._data_model.get_dataframe())
        slice_ratios = ratios / sum(ratios)
        slice_sizes = np.round(slice_ratios * data_length)

        results = []
        prev_breakpoint = 0
        for slice_size in slice_sizes:
            break_number = prev_breakpoint + int(slice_size)
            model_data_slice = model_data.iloc[prev_breakpoint:break_number, :]
            prev_breakpoint = break_number

            model_copy = deepcopy(self._data_model)
            model_copy.set_dataframe(model_data_slice)
            results.append(model_copy)

        return results


class TestDataSetSplitter(unittest.TestCase):
    def setUp(self):
        self._data = {
            'target': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'feature_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'feature_2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }

        self._dataframe = pd.DataFrame(data=self._data)
        self._data_model = DataModel(self._dataframe)
        self._data_model.set_tf_feature_columns([
            tf.feature_column.numeric_column('feature_1'),
            tf.feature_column.numeric_column('feature_2')
        ])

        self._data_model.set_target_column('target')

    def test_split_data(self):
        splitter = DataSetSplitter(self._data_model)
        evaluation_data, train_data = splitter.split_by_ratio(ratios=[20, 80])

        train_features = train_data.get_feature_columns()
        train_target = train_data.get_target_column()

        eval_features = evaluation_data.get_feature_columns()
        eval_target = evaluation_data.get_target_column()

        self.assertEqual(len(train_target), 8)
        self.assertEqual(len(train_features), 8)
        self.assertEqual(len(eval_target), 2)
        self.assertEqual(len(eval_features), 2)


class DataLoader:
    _ml_data_model = DataModel
    _conserve_memory = False

    def __init__(self, conserve_memory: bool = False):
        """

        :param conserve_memory: bool
        """
        self._conserve_memory = conserve_memory

    def load_csv(self, path: str):
        """

        :param path: str
        """
        dataframe = pd.read_csv(path, low_memory=self._conserve_memory)
        self._ml_data_model = DataModel(dataframe)

    def filter_columns(self, columns: list):
        """

        :param columns: list
        """
        dataframe = self._ml_data_model.get_dataframe()
        dataframe = dataframe[columns]
        self._ml_data_model = DataModel(dataframe)

    def get_dataset(self):
        """

        :return: void
        """
        return self._ml_data_model

    def set_dataset(self, dataset: DataModel):
        """

        :param dataset: pd.DataFrame
        """
        self._ml_data_model = dataset


# class Converter:
#     _data_model: DataModel
#
#     def __init__(self, dataset: DataModel):
#         """
#
#         :param dataset:
#         """
#         self._data_model = dataset
#
#     def insert_new_column(self, new_column: str, convert_fn):
#         """
#
#         :param new_column:
#         :param convert_fn:
#         """
#         dataframe = self._data_model.get_dataframe()
#         dataframe[new_column] = dataframe.apply(convert_fn, axis=1)
#
#     def replace_missing_data_with_category(self, columns: list, missing_category_name: str):
#         self._data_model.validate_columns(columns)
#         dataframe = self._data_model.get_dataframe()
#         dataframe[columns] = dataframe[columns].fillna(missing_category_name)
#
#     def get_dataset(self):
#         return self._data_model


if __name__ == '__main__':
    unittest.main()
