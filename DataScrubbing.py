from abc import ABC, abstractmethod
from typing import List, Union, Optional

from AIBuilder.AIFactory import BalanceData
from AIBuilder.AIFactory.BalanceData import UnbalancedDataStrategy, UnbalancedDataStrategyFactory
from AIBuilder.AIFactory.FeatureColumnStrategies import FromListCategoryGrabber
from AIBuilder.AIFactory.Printing import ConsolePrintStrategy, FactoryPrinter
from AIBuilder.AIFactory.Specifications import DataTypeSpecification, NullSpecification, TypeSpecification, Describer, \
    PrefixedDictSpecification
from AIBuilder.Data import DataModel, MetaData
from currency_converter import CurrencyConverter
from datetime import datetime
import numpy as np
from scipy import stats
import pandas as pd

from AIBuilder.Summizer import TimeSummizer
from AIBuilder.SyntacticTools import AliasLoader


class Scrubber(ABC, Describer):

    @property
    @abstractmethod
    def scrubber_config_list(self):
        """Returns dict of columns names with required types.

        example of output: {'value': 'numerical', 'type': 'categorical'}
        """

        pass

    @abstractmethod
    def validate(self, data_model: DataModel):
        """Validate data model before scrubbing."""
        pass

    def validate_metadata(self, meta_data: MetaData):
        self._validate_column_config_dict(meta_data)

    @abstractmethod
    def update_metadata(self, meta_data: MetaData):
        """Update data models meta data to emulate scrubbing data modifications."""
        pass

    @abstractmethod
    def scrub(self, data_model: DataModel) -> DataModel:
        pass

    def _validate_column_config_dict(self, meta_data: MetaData):
        for column, data_type in self.scrubber_config_list.items():
            if data_type is not meta_data.get_column_type(column):
                raise RuntimeError('scrubber {} validation: column {} should be of data type {}, type {} found'
                                   .format(self.__class__, column, data_type, meta_data.get_column_type(column)))


class ConvertToNumericScrubber(Scrubber):
    """ Convert series to numeric.
    """

    def __init__(self, column_names: List[str], downcast=None, errors='coerce'):
        self.errors = DataTypeSpecification('CTNS_errors', errors, str)
        self.column_names = DataTypeSpecification('CTNS_column_names', column_names, list)
        self.downcast = DataTypeSpecification('CSTNS_downcast', downcast, str)

    @property
    def scrubber_config_list(self):
        return {}

    def validate(self, data_model: DataModel):
        pass

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()
        for column_name in self.column_names():
            num_column = pd.to_numeric(df[column_name], downcast=self.downcast(), errors=self.errors())
            df[column_name] = num_column

        data_model.set_dataframe(df)

        return data_model


class MissingDataScrubber(Scrubber):

    def __init__(self, scrub_columns: List[str]):
        """ Removes rows with missing data.

        :param scrub_columns:
        """
        self.scrub_columns = DataTypeSpecification('MDS_scrub_columns', scrub_columns, list)

    @property
    def scrubber_config_list(self):
        return {}

    def validate(self, data_model: DataModel):
        pass

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()

        df = self.scrub_numerical(data_model, df)
        df = df.reset_index(drop=True)
        df = self.scrub_categorical(data_model, df)

        data_model.set_dataframe(df)

        return data_model

    def scrub_categorical(self, data_model, df):
        categorical_col_names_to_check = list(set(self.scrub_columns()) & set(data_model.metadata.categorical_columns))
        categorical_cols_to_check = df[categorical_col_names_to_check]
        indexes_with_none = self.get_indexes_with_none_value(categorical_cols_to_check)

        return df.drop(index=indexes_with_none)

    def scrub_numerical(self, data_model, df):
        numerical_col_names_to_check = list(set(self.scrub_columns()) & set(data_model.metadata.numerical_columns))
        numerical_cols_to_check = df[numerical_col_names_to_check]
        indexes_with_nan = self.get_indexes_with_nan_value(numerical_cols_to_check)

        return df.drop(index=indexes_with_nan)

    def get_indexes_with_nan_value(self, cols_to_check) -> set:
        indexes_with_nan = set()
        for index in cols_to_check.index:
            slice = cols_to_check.values[index]
            if self.has_nan(slice):
                indexes_with_nan.add(index)

        return indexes_with_nan

    def get_indexes_with_none_value(self, cols_to_check) -> set:
        indexes_with_none = set()
        for index in cols_to_check.index:
            slice = cols_to_check.values[index]
            if self.has_none(slice):
                indexes_with_none.add(index)

        return indexes_with_none

    @staticmethod
    def has_nan(values_to_check: list) -> bool:
        for value in values_to_check:
            if np.isnan(value):
                return True
        return False

    @staticmethod
    def has_none(values_to_check: list) -> bool:
        for value in values_to_check:
            # todo: change obj structure as is does no longer only check none.
            if value is None or type(value) is not str and np.isnan(value):
                return True
        return False


class MissingDataReplacer(Scrubber):

    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, scrub_columns: list, missing_category_name: str = 'unknown',
                 missing_numerical_value: Optional[Union[int, float, str]] = None):
        """ Scrubs missing data from dataset in various ways.

        :type missing_numerical_value: int|float|str
        :param missing_category_name: Replace missing values in categorical data to this value.
        :param scrub_columns: Scrub these columns.
        """
        self.missing_value = missing_numerical_value
        self.columns_to_scrub = scrub_columns
        self.missing_category_name = missing_category_name

    def validate(self, data_model: DataModel):
        pass

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        data_model = self.scrub_categorical_data(data_model)
        data_model = self.scrub_numerical_columns(data_model)

        return data_model

    def scrub_categorical_data(self, data_model):
        categorical_columns = data_model.metadata.categorical_columns
        categorical_columns_to_scrub = list(set(categorical_columns) & set(self.columns_to_scrub))
        data_model = self._scrub_categorical_data(data_model, categorical_columns_to_scrub)

        return data_model

    def scrub_numerical_columns(self, data_model):
        numerical_columns = data_model.metadata.numerical_columns
        numerical_columns_to_scrub = list(set(numerical_columns) & set(self.columns_to_scrub))
        data_model = self._scrub_numerical_data(data_model, numerical_columns_to_scrub)

        return data_model

    def _scrub_categorical_data(self, data_model: DataModel, categorical_columns: list) -> DataModel:
        data_model.validate_columns(categorical_columns)
        self.fillNaForColumns(data_model=data_model, columns=categorical_columns,
                              replacement_value=self.missing_category_name)

        return data_model

    def _scrub_numerical_data(self, data_model: DataModel, numerical_columns: list) -> DataModel:
        data_model.validate_columns(numerical_columns)

        if type(self.missing_value) is int and type(self.missing_value):
            self.fillNaForColumns(data_model, numerical_columns, self.missing_value)

        if self.missing_value == 'average':
            df = data_model.get_dataframe()
            for col in numerical_columns:
                average = self.get_column_average(data_model=data_model, column=col)
                df[col] = df[col].fillna(average)

            data_model.set_dataframe(df)

        return data_model

    @staticmethod
    def get_column_average(data_model: DataModel, column: str):
        df = data_model.get_dataframe()
        num_values = df[column].tolist()
        cleaned_num_values = [value for value in num_values if str(value) != 'nan']
        average = sum(cleaned_num_values) / len(cleaned_num_values)

        return average

    @staticmethod
    def fillNaForColumns(data_model: DataModel, columns: list, replacement_value):
        df = data_model.get_dataframe()
        df[columns] = df[columns].fillna(replacement_value)
        data_model.set_dataframe(df)


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

    def describe(self):
        description = {'self': super(AndScrubber, self).describe()}
        for scrubber in self.scrubber_list:
            description[scrubber.__class__.__name__] = scrubber.describe()

        return description

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


class BlackListScrubber(Scrubber):
    """ Removes rows with blacklisted categories.
    """

    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, column_name: str, black_list: List[str]):
        self.black_list = DataTypeSpecification('black_list', black_list, List[str])
        self.column_name = DataTypeSpecification('column_name', column_name, str)

    def validate(self, data_model: DataModel):
        df = data_model.get_dataframe()

        assert MetaData.CATEGORICAL_DATA_TYPE == data_model.metadata.get_column_type(
            self.column_name()), 'Column {} is not a categorical column.'.format(self.column_name())

        assert self.column_name() in df.columns, 'column name {} not in de dataframe'.format(self.column_name())

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()
        df = df.set_index(self.column_name())
        df = df.drop(self.black_list(), axis=0)
        df = df.reset_index()
        data_model.set_dataframe(df)

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
            self.all_z = DataTypeSpecification(name='OS_all_z', value=all_z, data_type=int)

        self.col_z = NullSpecification(name='OS_column_z_values')
        if col_z is not None:
            self.col_z = DataTypeSpecification(name='OS_column_z_values', value=col_z, data_type=dict)
            self.col_z = PrefixedDictSpecification(name='OS_column_z_values', prefix='OS_', value=col_z)

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


# tensor flow estimator cannot handle categorical dtypes.
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

        note: Use the 'MultipleCatToListScrubber' to create the lists in the input column.
    """

    def __init__(self, col_name: str, exclusion_list: List[str] = None, inclusion_threshold: float = None):
        """

        :param col_name:        column name (must be MULTIPLE_CAT_DATA_TYPE) and will be transformed into MULTIPLE_HOT_DATA_TYPE)
        :param exclusion_list:  categories to exclude
        """
        # todo: replace this with event dispatching system for dispatching console messages etc.
        self.time_summizer = TimeSummizer()
        self.col_name = DataTypeSpecification('col_name', col_name, str)
        self.exclusion_list = NullSpecification('exclusion_service')
        self.bin_cat_map = {}  # keeps track of {category: binary category name}.
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

        # todo takes very long time! 30+ sec!
        self.time_summizer.start_time_log()
        m_hot_data = self.fill_new_data_set(categories, inputColumn, m_hot_data)
        self.time_summizer.log('filled dataset', None)
        self.time_summizer.summize(ConsolePrintStrategy())
        self.time_summizer.reset()
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

    def get_new_data_set(self, categories: list):
        m_hot_data = {}
        for cat in categories:
            binary_cat_name = self.get_binary_cat_name(cat)
            self.bin_cat_map[cat] = binary_cat_name
            m_hot_data[binary_cat_name] = []

        return m_hot_data

    def fill_new_data_set(self, categories: List[str], input_column: pd.Series, m_hot_data: dict):
        data = input_column.to_dict().values()
        for item_categories in data:
            for category in categories:
                occurrences = list(item_categories).count(category)  # maybe just check if category is present
                binary_column_name = self.bin_cat_map[category]
                if occurrences > 0:
                    m_hot_data[binary_column_name].append(1)
                    continue

                m_hot_data[binary_column_name].append(0)

        return m_hot_data

    def get_binary_cat_name(self, category_name: str) -> str:
        bin_cat_name = self.col_name() + '_' + category_name
        bin_cat_name = bin_cat_name.replace(' ', '_')
        bin_cat_name = bin_cat_name.replace("'", '')

        return bin_cat_name

    @staticmethod
    def meets_inclusion_threshold(binary_data: list, min_threshold: Union[None, int], data_length: int):
        if min_threshold is None:
            return True

        positive_count = binary_data.count(1)
        max_threshold = data_length - min_threshold

        if min_threshold <= positive_count <= max_threshold:
            return True

        return False


class ConvertToColumnScrubber(Scrubber):
    """Create a new column based and have it filled by callable.
    """

    def __init__(self, new_column_name: str, new_column_type: str, converter: callable, required_columns: dict):
        self.new_column_type = new_column_type
        self.required_columns = required_columns
        self.converter = converter
        self.new_column_name = new_column_name

    @property
    def scrubber_config_list(self):
        if None is not self.required_columns:
            return self.required_columns

        return {}

    def validate(self, data_model: DataModel):
        for column, data_type in self.required_columns.items():
            assert column in data_model.get_dataframe()
            assert data_type is data_model.metadata.get_column_type(column), \
                'failed asserting that column: {} is of type: {} from metadata'.format(column, data_type)

    def update_metadata(self, meta_data: MetaData):
        meta_data.add_column_to_type(column_name=self.new_column_name, column_type=self.new_column_type)

        return meta_data

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()
        df[self.new_column_name] = df.apply(self.converter, axis=1)
        data_model.set_dataframe(df)

        return data_model


class CategoryToFloatScrubber(ConvertToColumnScrubber):
    """Add a column with a float value based on categorical column"""

    def __init__(self, new_column_name: str, source_column_name: str, category_to_value_index: dict):
        def convert(row):
            cat = row[source_column_name]
            return category_to_value_index[cat]

        super().__init__(new_column_name=new_column_name,
                         new_column_type=MetaData.NUMERICAL_DATA_TYPE,
                         converter=convert,
                         required_columns={source_column_name: MetaData.CATEGORICAL_DATA_TYPE})


class ColumnBinner(Scrubber):

    def __init__(self, source_column_name: str, target_column_name: str, bins: list, labels: list):
        self.source_column_name = DataTypeSpecification('source_column_name', source_column_name, str)
        self.target_column_name = DataTypeSpecification('target_column_name', target_column_name, str)
        self.bins = DataTypeSpecification('bins', bins, list)
        self.labels = DataTypeSpecification('labels', labels, list)

    @property
    def scrubber_config_list(self):
        return {self.source_column_name(): MetaData.NUMERICAL_DATA_TYPE}

    def validate(self, data_model: DataModel):
        assert self.source_column_name() in data_model.get_dataframe()

    def update_metadata(self, meta_data: MetaData):
        meta_data.add_column_to_type(column_name=self.target_column_name(), column_type=MetaData.CATEGORICAL_DATA_TYPE)

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()
        df[self.target_column_name()] = pd.cut(df[self.source_column_name()], bins=self.bins(), labels=self.labels())
        data_model.set_dataframe(df)

        return data_model


class KeyWordToCategoryScrubber(ConvertToColumnScrubber):
    """ Search for keyword in text to determine category.
    """

    # todo make this multi cat later.

    def __init__(self, new_column_name: str, source_column_name: str, keywords: List[str], unknown_category: str,
                 use_synonyms: Optional[bool] = False, min_syntactic_distance: Optional[float] = None,
                 verbose: Optional[int] = 0):
        """
        :param new_column_name:
        :param source_column_name:
        :param keywords: list of keywords as values.
        :param unknown_category:
        :param use_synonyms: if True synonyms to the keywords will be used to determine category
        :param min_syntactic_distance: minimum similarity between keyword and its synonym for the synonym to be used.
        :param verbose: bool, 0: not verbose, 1: verbose, 2: verbose
        """
        self.verbosity = verbose
        self.min_syntactic_distance = DataTypeSpecification('min syntactic distance', 0.0, float)
        if min_syntactic_distance is not None:
            self.min_syntactic_distance.value = min_syntactic_distance

        self.use_synonyms = DataTypeSpecification('use synonyms', use_synonyms, bool)

        alias_loader = AliasLoader(use_synonyms=self.use_synonyms(),
                                   min_syntactic_distance=self.min_syntactic_distance(),
                                   verbosity=self.verbosity)

        cat_alias = alias_loader.load_cat_aliases(keywords)

        def convert(row):
            category = unknown_category
            stop = False
            for cat, aliases in cat_alias.items():
                for alias in aliases:
                    if stop is True:
                        break

                    row_value = row[source_column_name]
                    if alias.lower() + ' ' in row_value.lower() + ' ':
                        if self.verbosity > 1:
                            print('value "{}", considered of cat "{}", through alias: "{}".'.format(row_value, cat,
                                                                                                    alias))

                        category = cat
                        stop = True

            return category

        super().__init__(new_column_name=new_column_name,
                         new_column_type=MetaData.CATEGORICAL_DATA_TYPE,
                         converter=convert,
                         required_columns={source_column_name: MetaData.TEXT_DATA_TYPE})


class BinaryResampler(Scrubber):
    """ Resample an imbalanced dataset by a binary category. If 70% of data points is positive, for example, the dataset
        is imbalanced. This scrubber can either copy the negative data points until the dataset is balanced or remove
        negative data points.
    """

    WEIGHT_COLUMN = BalanceData.WEIGHTS_COLUMN

    def __init__(self, column_name: str, strategy: str, shuffle: bool = True):
        self.column_name = DataTypeSpecification('BR_column_names', column_name, str)
        self.strategy = TypeSpecification('BR_strategy', strategy,
                                          [UnbalancedDataStrategy.OVER_SAMPLING, UnbalancedDataStrategy.UNDER_SAMPLING,
                                           UnbalancedDataStrategy.RE_WEIGH])
        self.shuffle = DataTypeSpecification('BR_re-shuffle', shuffle, bool)
        self.factory = UnbalancedDataStrategyFactory()

    @property
    def scrubber_config_list(self):
        return {self.column_name(): MetaData.CATEGORICAL_DATA_TYPE}

    def validate(self, data_model: DataModel):
        df = data_model.get_dataframe()
        unique_categories = df[self.column_name()].unique()

        assert len(unique_categories) == 2, \
            'Column {} is not binary, categories found: {}'.format(self.column_name(), str(unique_categories))

        assert self.column_name() in df.columns
        assert self.column_name() in data_model.metadata.categorical_columns

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        strategy = self.factory.get_strategy(self.strategy())
        data_model = strategy.balance_data(data_model=data_model, target_column_name=self.column_name())
        data_model.set_dataframe(self.shuffle_data(scrubbed_df=data_model.get_dataframe()))

        return data_model

    def shuffle_data(self, scrubbed_df: pd.DataFrame) -> pd.DataFrame:
        if self.shuffle():
            # give sample a fixed random state to make results reproducible.
            scrubbed_df = scrubbed_df.sample(frac=1, random_state=1)
        scrubbed_df = scrubbed_df.reset_index(drop=True)

        return scrubbed_df
