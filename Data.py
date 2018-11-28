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

    def __init__(self, data: pd.DataFrame = None):

        self.uncategorized_columns = list()

        if data is not None:
            self.uncategorized_columns = list(data)

        self.categorical_columns = list()
        self.numerical_columns = list()

    def __repr__(self):
        return repr({
            'categorical': self.categorical_columns,
            'numerical': self.numerical_columns,
            'unknown': self.uncategorized_columns
        })

    def __str__(self):
        stringy_self = '\nmetadata:'

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
        assert column in self._dataframe, 'Column {} not in data frame.'.format(column)

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
        self.feature_columns_names = feature_column_names

    def get_feature_columns(self):
        self.validate_columns(self.feature_columns_names)

        return self._dataframe[self.feature_columns_names]

    def set_target_column(self, target_column_name: str):
        self.target_column_name = target_column_name

    def get_target_column(self):
        self.validate_columns([self.target_column_name])

        return self._dataframe[self.target_column_name]

    def __repr__(self):
        return repr({
            'data_model': self.metadata,
            'source': 'todo',
            'scrubbing': 'todo'
        })


class DataSetSplitter:
    _data_model: DataModel

    def __init__(self, data_model: DataModel):
        self._data_model = data_model

    def split_by_ratio(self, ratios: list) -> list:
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


class DataLoader:
    _ml_data_model = DataModel
    _conserve_memory = False

    def load_csv(self, path: str):
        dataframe = pd.read_csv(path)
        self._ml_data_model = DataModel(dataframe)

    def filter_columns(self, columns: list):
        dataframe = self._ml_data_model.get_dataframe()
        columns = [column for column in columns if column in dataframe.columns]
        dataframe = dataframe[columns]
        self._ml_data_model = DataModel(dataframe)

    def get_dataset(self):

        return self._ml_data_model

    def set_dataset(self, dataset: DataModel):
        self._ml_data_model = dataset
