from abc import ABC, abstractmethod
from typing import List, Union
from AIBuilder.AIFactory.FeatureColumnStrategies import FromListCategoryGrabber
from AIBuilder.AIFactory.Printing import ConsolePrintStrategy, FactoryPrinter
from AIBuilder.AIFactory.Specifications import DataTypeSpecification, NullSpecification
from AIBuilder.Data import DataModel, MetaData
from currency_converter import CurrencyConverter
from datetime import datetime
import numpy as np
from scipy import stats
import pandas as pd


class Scrubber(ABC):

    @property
    @abstractmethod
    def scrubber_config_list(self):
        pass

    @abstractmethod
    def validate(self, data_model: DataModel):
        pass

    def validate_metadata(self, meta_data: MetaData):
        self._validate_column_config_dict(meta_data)

    @abstractmethod
    def update_metadata(self, meta_data: MetaData):
        pass

    @abstractmethod
    def scrub(self, data_model: DataModel) -> DataModel:
        pass

    def _validate_column_config_dict(self, meta_data: MetaData):
        for column, data_type in self.scrubber_config_list.items():
            if data_type is not meta_data.get_column_type(column):
                raise RuntimeError('scrubber {} validation: column {} should be of data type {}, type {} found'
                                   .format(self.__class__, column, data_type, meta_data.get_column_type(column)))


class MissingDataScrubber(Scrubber):

    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, missing_category_name: str):
        self._missing_category_name = missing_category_name

    def validate(self, data_model: DataModel):
        pass

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        categorical_columns = data_model.metadata.categorical_columns

        return self._scrub_categorical_data(data_model, categorical_columns)

    def _scrub_categorical_data(self, data_model: DataModel, categorical_columns: list) -> DataModel:
        data_model.validate_columns(categorical_columns)
        df = data_model.get_dataframe()
        df[categorical_columns] = df[categorical_columns].fillna(self._missing_category_name)
        data_model.set_dataframe(df)

        return data_model


class StringToDateScrubber(Scrubber):

    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, date_columns: dict, ):
        self.date_columns = date_columns

    def validate(self, data_model: DataModel):
        for date_column, format in self.date_columns.items():
            data_model.validate_columns([date_column])

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()

        for date_column, format in self.date_columns.items():
            split_format = format.split('T')
            date_format = split_format[0]

            def convert(value):
                return datetime.strptime(
                    value[date_column].split('T')[0],
                    date_format
                )

            df[date_column] = df.apply(convert, axis=1)


class AverageColumnScrubber(Scrubber):

    @property
    def scrubber_config_list(self):
        required_column_config = {self.new_average_column: None}

        for input_column in self._input_columns:
            required_column_config[input_column] = MetaData.NUMERICAL_DATA_TYPE

        return required_column_config

    def __init__(self, input_columns: tuple, output_column: str):
        self._input_columns = input_columns
        self.new_average_column = output_column

    def validate(self, data_model: DataModel):
        data_model.validate_columns(self._input_columns)

    def update_metadata(self, meta_data: MetaData):
        meta_data.define_numerical_columns([self.new_average_column])

        return meta_data

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()
        df[self.new_average_column] = 0

        for input_column in self._input_columns:
            df[self.new_average_column] = df[self.new_average_column] + df[input_column]

        df[self.new_average_column] = df[self.new_average_column] / len(self._input_columns)
        data_model.set_dataframe(df)

        return data_model


class ConvertCurrencyScrubber(Scrubber):

    @property
    def scrubber_config_list(self):
        required_column_config = {self.from_currency_column: MetaData.CATEGORICAL_DATA_TYPE,
                                  self.value_column: MetaData.NUMERICAL_DATA_TYPE}

        if self.exchange_rate_date_column:
            required_column_config[self.exchange_rate_date_column] = MetaData.UNKNOWN_DATA_TYPE

        return required_column_config

    def __init__(self, value_column: str, new_value_column: str, from_currency_column: str, to_currency: str,
                 original_date_column: str = None):

        self.value_column = value_column
        self.new_value_column = new_value_column
        self.from_currency_column = from_currency_column
        self.to_currency = to_currency
        self.exchange_rate_date_column = original_date_column
        self.converter = CurrencyConverter(fallback_on_missing_rate=True, fallback_on_wrong_date=True)

    def validate(self, data_model: DataModel):
        required_columns = [self.value_column, self.from_currency_column]

        if self.exchange_rate_date_column is not None:
            required_columns.append(self.exchange_rate_date_column)

        data_model.validate_columns(required_columns)

    def update_metadata(self, meta_data: MetaData):
        if meta_data.get_column_type(self.new_value_column) is None:
            meta_data.define_numerical_columns([self.new_value_column])

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()

        def convert_func(value):
            date_argument = None
            if self.exchange_rate_date_column is not None:
                date_argument = value[self.exchange_rate_date_column]

            return self.converter.convert(value[self.value_column],
                                          value[self.from_currency_column],
                                          new_currency=self.to_currency,
                                          date=date_argument)

        df[self.new_value_column] = df.apply(convert_func, axis=1)
        data_model.set_dataframe(df)

        return data_model


class AndScrubber(Scrubber):
    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, *scrubbers: Scrubber):
        # todo: replace this with event dispatching system for dispatching console messages etc.
        self.console_printer = FactoryPrinter(ConsolePrintStrategy())
        self.scrubber_list = []
        for scrubber in scrubbers:
            self.add_scrubber(scrubber)

    def add_scrubber(self, scrubber: Scrubber):
        self.scrubber_list.append(scrubber)

    def validate(self, data_model: DataModel):
        pass

    def validate_metadata(self, meta_data: MetaData):
        for scrubber in self.scrubber_list:
            scrubber.validate_metadata(meta_data)
            scrubber.update_metadata(meta_data)

    def update_metadata(self, meta_data: MetaData):
        for scrubber in self.scrubber_list:
            scrubber.update_metadata(meta_data)

    def scrub(self, data_model: DataModel) -> DataModel:
        for scrubber in self.scrubber_list:
            scrubber.validate(data_model)
            scrubber.update_metadata(meta_data=data_model.metadata)
            self.console_printer.line('  - Scrubbing: ' + scrubber.__class__.__name__)
            scrubber.scrub(data_model)

        return data_model


class OutlierScrubber(Scrubber):

    @property
    def scrubber_config_list(self):
        required_columns = {}

        if False is isinstance(self.col_z, NullSpecification):
            for column, z in self.col_z().items():
                required_columns[column] = MetaData.NUMERICAL_DATA_TYPE

        return required_columns

    def __init__(self, col_z: dict = None, all_z: int = None):
        self.all_z = NullSpecification(name='all_z')
        if all_z is not None:
            self.all_z = DataTypeSpecification(name='all_z', value=all_z, data_type=int)

        self.col_z = NullSpecification(name='column_z_values')
        if col_z is not None:
            self.col_z = DataTypeSpecification(name='column_z_values', value=col_z, data_type=dict)

        assert col_z is None or all_z is None, 'coll_z and all_z cannot be used together in outlier scrubber.'
        assert col_z is not None or all_z is not None, 'Either coll_z or all_z is required in outlier scrubber.'

    def validate(self, data_model: DataModel):
        if isinstance(self.col_z, NullSpecification):
            return

        for column, z in self.col_z().items():
            column_data_type = data_model.metadata.get_column_type(column=column)
            assert MetaData.NUMERICAL_DATA_TYPE == column_data_type, \
                '{} column passed to outlier scrubber not numerical but {}.'.format(column, column_data_type)

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()
        if False is isinstance(self.col_z, NullSpecification):
            column_z_values = self.col_z()
            for column, z in column_z_values.items():
                df = self.remove_outlier_from_df_column(column, df, z)

        if False is isinstance(self.all_z, NullSpecification):
            for column in data_model.metadata.numerical_columns:
                df = self.remove_outlier_from_df_column(column, df, self.all_z())

        data_model.set_dataframe(df)

        return data_model

    @staticmethod
    def remove_outlier_from_df_column(column, df, z):
        df = df[np.abs(stats.zscore(df[column])) < z]
        return df


class MakeCategoricalScrubber(Scrubber):
    @property
    def scrubber_config_list(self):
        return {}

    def validate(self, data_model: DataModel):
        pass

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()
        metaData = data_model.metadata
        for column in metaData.categorical_columns:
            df[column] = df[column].astype('category')

        data_model.set_dataframe(df)

        return data_model


class MultipleCatToListScrubber(Scrubber):
    """ Turns MULTIPLE_CAT_COLUMNS data columns from string to list.

        before {'column': 'cat1, cat2, cat3', 'cat1, cat2, cat4'}
        after {'column': ['cat1', 'cat2', 'cat3'], ['cat1', 'cat2', 'cat4']}

    """

    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, sepperator: str):
        """

        :param sepperator:  Sub string that separates categories in data.
        """
        self.sepperator = sepperator

    def validate(self, data_model: DataModel):
        pass

    def update_metadata(self, meta_data: MetaData):
        pass

    @staticmethod
    def process(data):
        new_data = []
        for row in data:
            new_row = []
            for cat in row:
                cat = cat.replace(' ', '_')
                cat = cat.replace("'", '')
                new_row.append(cat)
            new_data.append(new_row)

        return new_data

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()
        metaData = data_model.metadata
        for column in metaData.multiple_cat_columns:
            df[column] = df[column].str.split(self.sepperator)
            df[column] = self.process(df[column])

        return data_model.set_dataframe(df)


class MultipleCatListToMultipleHotScrubber(Scrubber):
    """ Turns multiple cat list to multiple hot in df.

        before {'column': ['cat1', 'cat2', 'cat3'], ['cat1', 'cat2', 'cat4']}
        after  {'column_cat1': [1, 1]
                'column_cat2': [1, 1]
                'column_cat3': [1, 0]
                'column_cat4': [0, 1]}
    """

    def __init__(self, col_name: str, exclusion_list: List[str] = None, inclusion_threshold: float = None):
        """

        :param col_name:        column name (must be MULTIPLE_CAT_DATA_TYPE) and will be transformed into MULTIPLE_HOT_DATA_TYPE)
        :param exclusion_list:  categories to exclude
        """
        self.col_name = DataTypeSpecification('col_name', col_name, str)
        self.exclusion_list = NullSpecification('exclusion_service')
        if exclusion_list is not None:
            self.exclusion_list = DataTypeSpecification('exclusion_list', exclusion_list, list)

        self.inclusion_threshold = NullSpecification('inclusion_threshold')
        if inclusion_threshold is not None:
            self.inclusion_threshold = DataTypeSpecification('inclusion_threshold', inclusion_threshold, float)

    @property
    def scrubber_config_list(self):
        return {self.col_name(): MetaData.MULTIPLE_CAT_DATA_TYPE}

    def validate(self, data_model: DataModel):
        assert self.col_name() in data_model.get_dataframe()
        column_type = data_model.metadata.get_column_type(self.col_name())
        assert MetaData.MULTIPLE_CAT_DATA_TYPE == column_type, 'found that columns {} is type {} instead of multiple_' \
                                                               'cat.'.format(self.col_name(), column_type)

    def update_metadata(self, meta_data: MetaData):
        meta_data.remove_column(self.col_name())
        meta_data.add_column_to_type(column_name=self.col_name(), column_type=MetaData.MULTIPLE_HOT_DATA_TYPE)

    def scrub(self, data_model: DataModel) -> DataModel:
        inputColumn = self.get_input_column(data_model)

        categories = self.get_categories(data_model)
        m_hot_data = self.get_new_data_set(categories)
        m_hot_data = self.fill_new_data_set(categories, inputColumn, m_hot_data)

        data_model = self.add_binary_data_to_model(data_model, m_hot_data)
        data_model = self.update_metadata_on_scrub(data_model, m_hot_data)

        return data_model

    def get_input_column(self, data_model):
        df = data_model.get_dataframe()
        inputColumn = df[self.col_name()]

        return inputColumn

    @staticmethod
    def update_metadata_on_scrub(data_model, m_hot_data):
        binary_columns = list(m_hot_data.keys())
        metadata = data_model.metadata
        metadata.define_binary_columns(binary_columns)
        data_model.metadata = metadata

        return data_model

    def add_binary_data_to_model(self, data_model: DataModel, m_hot_data: dict) -> DataModel:
        df = data_model.get_dataframe()
        data_length = len(df)
        nominal_threshold = None
        if self.inclusion_threshold() is not None:
            nominal_threshold = self.inclusion_threshold() * len(df)

        for binary_column_name, binary_data in m_hot_data.items():
            if self.meets_inclusion_threshold(binary_data, nominal_threshold, data_length):
                data_model.add_feature_column(new_feature_column_name=binary_column_name)
                df[binary_column_name] = binary_data

        data_model.set_dataframe(df)

        return data_model

    def get_categories(self, data_model):
        categories_grabber = FromListCategoryGrabber(
            data_model=data_model,
            column_name=self.col_name(),
            exclusion_list=self.exclusion_list())
        categories = categories_grabber.grab()

        return categories

    def fill_new_data_set(self, categories: List[str], input_column: pd.Series, m_hot_data: dict):
        data = input_column.to_dict().values()
        for item_categories in data:
            for category in categories:
                occurrences = list(item_categories).count(category)
                # todo centralize column name rendering and make sure it matches regex: https://regex101.com/r/7yYIde/1
                binary_column_name = self.get_binary_cat_name(category)
                if occurrences > 0:
                    m_hot_data[binary_column_name].append(1)
                    continue

                m_hot_data[binary_column_name].append(0)

        return m_hot_data

    def get_binary_cat_name(self, category) -> str:
        bin_cat = self.col_name() + '_' + category
        bin_cat = bin_cat.replace(' ', '_')
        bin_cat = bin_cat.replace("'", '')

        return bin_cat

    def get_new_data_set(self, categories: list):
        m_hot_data = {}
        for cat in categories:
            binary_cat_name = self.get_binary_cat_name(cat)
            m_hot_data[binary_cat_name] = []

        return m_hot_data

    @staticmethod
    def meets_inclusion_threshold(binary_data: list, min_threshold: Union[None, int], data_length: int):
        if min_threshold is None:
            return True

        positive_count = binary_data.count(1)
        max_threshold = data_length - min_threshold

        if min_threshold <= positive_count <= max_threshold:
            return True

        return False
