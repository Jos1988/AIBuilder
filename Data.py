import pandas as pd
import numpy as np
from copy import deepcopy
from typing import Sequence, Optional, List


class MetaData:
    CATEGORICAL_DATA_TYPE = 'categorical'
    NUMERICAL_DATA_TYPE = 'numerical'
    MULTIPLE_CAT_DATA_TYPE = 'multiple_cat'
    MULTIPLE_HOT_DATA_TYPE = 'multiple_hot'
    BINARY_DATA_TYPE = 'binary'
    TEXT_DATA_TYPE = 'text'
    UNKNOWN_DATA_TYPE = 'unknown'

    def __init__(self, data: pd.DataFrame = None):

        self.uncategorized_columns = list()

        if data is not None:
            self.uncategorized_columns = list(data)

        self.multiple_cat_columns = list()
        self.multiple_hot_columns = list()
        self.categorical_columns = list()
        self.numerical_columns = list()
        self.text_columns = list()
        self.binary_columns = list()

        self.column_collections = {self.MULTIPLE_CAT_DATA_TYPE: self.multiple_cat_columns,
                                   self.MULTIPLE_HOT_DATA_TYPE: self.multiple_hot_columns,
                                   self.CATEGORICAL_DATA_TYPE: self.categorical_columns,
                                   self.NUMERICAL_DATA_TYPE: self.numerical_columns,
                                   self.UNKNOWN_DATA_TYPE: self.uncategorized_columns,
                                   self.BINARY_DATA_TYPE: self.binary_columns,
                                   self.TEXT_DATA_TYPE: self.text_columns}

    def __repr__(self):
        self.sort_column_lists()

        return repr({
            'categorical': self.categorical_columns,
            'multiple_cat': self.multiple_cat_columns,
            'multiple_hot': self.multiple_hot_columns,
            'numerical': self.numerical_columns,
            'binary': self.binary_columns,
            'text': self.text_columns,
            'unknown': self.uncategorized_columns
        })

    def __str__(self):
        self.sort_column_lists()
        stringy_self = '\nmetadata:'

        for column_type, collection in self.column_collections.items():
            for column in collection:
                stringy_self = stringy_self + '\n' + column + ': ' + column_type

        return stringy_self

    def sort_column_lists(self):
        for column_type, collection in self.column_collections.items():
            collection.sort()

    def define_categorical_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.CATEGORICAL_DATA_TYPE, column_name)

    def define_numerical_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.NUMERICAL_DATA_TYPE, column_name)

    def define_unknown_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.UNKNOWN_DATA_TYPE, column_name)

    def define_multiple_cat_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.MULTIPLE_CAT_DATA_TYPE, column_name)

    def define_multiple_hot_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.MULTIPLE_HOT_DATA_TYPE, column_name)

    def define_binary_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.BINARY_DATA_TYPE, column_name)

    def define_text_columns(self, column_names: list):
        for column_name in column_names:
            self.add_column_to_type(self.TEXT_DATA_TYPE, column_name)

    def add_column_to_type(self, column_type: str, column_name: str):
        self.remove_column(column_name)
        collection = self.column_collections[column_type]
        if column_name not in collection:
            collection.append(column_name)

    def remove_column(self, column: str):
        for column_type, collection in self.column_collections.items():
            if column in collection:
                collection.remove(column)

    def get_column_type(self, column: str) -> Optional[str]:
        for column_type, collection in self.column_collections.items():
            if column in collection:
                return column_type

        return None


class DataModel:
    _dataframe: pd.DataFrame
    _tf_feature_columns: list
    # todo: really use all these getters and setters?

    def __init__(self, data: pd.DataFrame):
        self.metadata = MetaData(data)
        self._dataframe = data
        # todo use metadata object instead?
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
        self._tf_feature_columns = feature_columns

    def get_tf_feature_columns(self):
        return self._tf_feature_columns

    def set_feature_columns(self, feature_column_names: List[str]):
        self.feature_columns_names = feature_column_names

    def add_feature_column(self, new_feature_column_name: str):
        self.feature_columns_names.append(new_feature_column_name)

    def get_feature_columns(self):
        self.validate_columns(self.feature_columns_names)

        return self._dataframe[self.feature_columns_names]

    def set_target_column(self, target_column_name: str):
        self.target_column_name = target_column_name

    def get_target_column(self):
        self.validate_columns([self.target_column_name])

        return self._dataframe[self.target_column_name]

    def __len__(self):
        return len(self._dataframe)

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

    def filter_columns(self, columns: set):
        dataframe = self._ml_data_model.get_dataframe()
        columns = [column for column in columns if column in dataframe.columns]
        dataframe = dataframe[columns]
        self._ml_data_model = DataModel(dataframe)

    def get_dataset(self):
        return self._ml_data_model

    def set_dataset(self, dataset: DataModel):
        self._ml_data_model = dataset
