import unittest
from abc import ABC, abstractmethod
from AIBuilder.Data import DataModel, MetaData
import pandas as pd
from currency_converter import CurrencyConverter
from datetime import datetime


# abstract class
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


class TestMissingDataScrubber(unittest.TestCase):

    def setUp(self):
        self._data = {
            'numerical_1': [1, 2, 3],
            'categorical_1': ['one', None, 'two'],
            'categorical_2': ['apple', 'pie', None],
            'unknown_1': [9, 10, 11]
        }

        self._dataframe = pd.DataFrame(data=self._data)
        self._data_model = DataModel(self._dataframe)

    def test_categorize_columns(self):
        categorical = ['categorical_1', 'categorical_2']
        self._data_model.metadata.define_categorical_columns(categorical)

        unknown_category = 'unknown'
        missing_data_scrubber = MissingDataScrubber(unknown_category)
        missing_data_scrubber.validate(self._data_model)
        missing_data_scrubber.scrub(self._data_model)
        self.assertEqual(unknown_category, self._data_model.get_dataframe()['categorical_1'][1])
        self.assertEqual(unknown_category, self._data_model.get_dataframe()['categorical_2'][2])
        self.assertEqual(1, self._data_model.get_dataframe()['numerical_1'][0])
        self.assertEqual(9, self._data_model.get_dataframe()['unknown_1'][0])


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


class TestDateScrubber(unittest.TestCase):

    def setUp(self):
        self._data = {
            'date': ['2017-03-26T05:04:46.539Z', '2017-12-01T23:04:46.539Z', '2017-02-08T07:38:48.129Z'],
        }

        self._dataframe = pd.DataFrame(data=self._data)
        self.data_model = DataModel(self._dataframe)

    def test_validate(self):
        data_scrubber = StringToDateScrubber(date_columns={'date': '%Y-%m-%d'})

        data_scrubber.validate(self.data_model)

    def test_scrub(self):
        data_scrubber = StringToDateScrubber(date_columns={'date': '%Y-%m-%d'})

        data_scrubber.scrub(self.data_model)
        date = datetime.strptime('2017-12-01', '%Y-%m-%d')
        self.assertEqual(date, self.data_model.get_dataframe()['date'][1])


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


class TestAverageColumnScrubber(unittest.TestCase):

    def setUp(self):
        self._data = {
            'column_1': [0, 2, 3],
            'column_2': [3, 2, 1],
            'column_3': [3, 2, 2],
        }

        self._dataframe = pd.DataFrame(data=self._data)
        self._data_model = DataModel(self._dataframe)
        self._data_model.metadata.define_numerical_columns(['column_1', 'column_2', 'column_3'])

    def test_invalid_column(self):
        input_columns = ('column_1', 'column_2', 'invalid_1')
        average_column = 'average'

        average_converter = AverageColumnScrubber(input_columns, average_column)

        with self.assertRaises(RuntimeError):
            average_converter.validate(self._data_model)

    def test_valid_metadata(self):
        input_columns = ('column_1', 'column_2', 'column_3')
        average_column = 'average'

        average_converter = AverageColumnScrubber(input_columns, average_column)
        self.assertIsNone(average_converter.validate_metadata(self._data_model.metadata))

    def test_invalid_metadata(self):
        input_columns = ('column_1', 'column_2', 'column_3')
        average_column = 'average'

        self._data_model.metadata.define_categorical_columns(['column_3'])
        average_converter = AverageColumnScrubber(input_columns, average_column)
        with self.assertRaises(RuntimeError):
            average_converter.validate_metadata(self._data_model.metadata)

    def test_averaging_3_columns(self):
        input_columns = ('column_1', 'column_2', 'column_3')
        average_column = 'average'

        average_converter = AverageColumnScrubber(input_columns, average_column)
        average_converter.validate(self._data_model)
        average_converter.scrub(self._data_model)

        self.assertListEqual([2, 2, 2], self._data_model.get_dataframe()[average_column].tolist())


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


class TestConvertCurrencyScrubber(unittest.TestCase):

    def setUp(self):
        date_1 = datetime.strptime('01-01-2018', '%d-%m-%Y')
        date_2 = datetime.strptime('01-01-2017', '%d-%m-%Y')
        date_3 = datetime.now()

        self._data = {
            'value': [0, 2, 100],
            'currency': ['EUR', 'USD', 'EUR'],
            'date': [date_3, date_1, date_2],
        }

        self._dataframe = pd.DataFrame(data=self._data)
        self._data_model = DataModel(self._dataframe)

    def test_valid_columns(self):
        currency_converter = ConvertCurrencyScrubber('value', 'currency', 'USD', 'date')

        with self.assertRaises(RuntimeError):
            currency_converter.validate(self._data_model)

    def test_valid_metadata(self):
        self._data_model.metadata.define_numerical_columns(['value'])
        self._data_model.metadata.define_categorical_columns(['currency'])

        currency_converter = ConvertCurrencyScrubber('value', 'value', 'currency', 'USD', 'date')
        currency_converter.validate_metadata(self._data_model.metadata)

    def test_valid_columns_without_date(self):
        currency_converter = ConvertCurrencyScrubber('value', 'value', 'currency', 'USD')
        currency_converter.validate(self._data_model)

    def test_invalid_column(self):
        currency_converter = ConvertCurrencyScrubber('value', 'invalid', 'USD', 'date')

        with self.assertRaises(RuntimeError):
            currency_converter.validate(self._data_model)

    def test_invalid_metadata(self):
        self._data_model.metadata.define_numerical_columns(['value', 'currency'])

        currency_converter = ConvertCurrencyScrubber('value', 'currency', 'USD', 'date')
        with self.assertRaises(RuntimeError):
            currency_converter.validate_metadata(self._data_model.metadata)

    def test_invalid_date_column(self):
        currency_converter = ConvertCurrencyScrubber('value', 'invalid', 'USD', 'invalid_date')

        with self.assertRaises(RuntimeError):
            self.assertEqual(False, currency_converter.validate(self._data_model))

    def test_converting_with_date_column(self):
        currency_converter = ConvertCurrencyScrubber('value', 'value', 'currency', 'USD', 'date')

        currency_converter.scrub(self._data_model)

        df = self._data_model.get_dataframe()
        result = df['value']
        self.assertEqual(0.0, result[0])
        self.assertEqual(2.0, result[1])
        self.assertEqual(105, round(result[2]))

    def test_converting_without_date_column(self):
        currency_converter = ConvertCurrencyScrubber('value', 'value', 'currency', 'USD')

        currency_converter.scrub(self._data_model)

        df = self._data_model.get_dataframe()
        result = df['value']
        self.assertEqual(0.0, result[0])
        self.assertEqual(2.0, result[1])
        self.assertEqual(116, round(result[2]))


class AndScrubber(Scrubber):
    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, *scrubbers: Scrubber):
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
            scrubber.scrub(data_model)

        return data_model


class TestAndScrubber(unittest.TestCase):
    def setUp(self):
        date_1 = datetime.strptime('01-01-2018', '%d-%m-%Y')
        date_2 = datetime.strptime('01-01-2017', '%d-%m-%Y')
        date_3 = datetime.now()

        self._data = {
            'value_1': [0, 1, 50],
            'value_2': [0, 3, 150],
            'currency': ['EUR', 'USD', 'EUR'],
            'date': [date_3, date_1, date_2],
        }

        self._df = pd.DataFrame(self._data)
        self.data_model = DataModel(self._df)
        self.data_model.metadata.define_numerical_columns(['value_1', 'value_2'])

        self.data_model.metadata.define_categorical_columns(['currency'])

    def test_multiple_validation(self):
        average_column_scrubber = AverageColumnScrubber(('value_1', 'value_2'), 'value')
        convert_currency_scrubber = ConvertCurrencyScrubber('value', 'new_value', 'currency', 'USD', 'date')
        self.and_scrubber = AndScrubber(average_column_scrubber, convert_currency_scrubber)

        self.and_scrubber.validate_metadata(self.data_model.metadata)

    def test_first_invalid(self):
        average_column_scrubber = AverageColumnScrubber(('value_1', 'value_2, value_3'), 'value')
        convert_currency_scrubber = ConvertCurrencyScrubber('value', 'currency', 'USD', 'date')
        self.and_scrubber = AndScrubber(average_column_scrubber, convert_currency_scrubber)

        with self.assertRaises(RuntimeError):
            self.and_scrubber.validate_metadata(self.data_model.metadata)

    def test_second_invalid(self):
        average_column_scrubber = AverageColumnScrubber(('value_1', 'value_2'), 'value')
        convert_currency_scrubber = ConvertCurrencyScrubber('value', 'invalid', 'USD', 'date')
        self.and_scrubber = AndScrubber(average_column_scrubber, convert_currency_scrubber)

        with self.assertRaises(RuntimeError):
            self.and_scrubber.validate_metadata(self.data_model.metadata)

    def test_multiple_scrubbing(self):
        average_column_scrubber = AverageColumnScrubber(('value_1', 'value_2'), 'value')
        convert_currency_scrubber = ConvertCurrencyScrubber('value', 'new_value', 'currency', 'USD', 'date')
        self.and_scrubber = AndScrubber(average_column_scrubber, convert_currency_scrubber)

        self.and_scrubber.scrub(self.data_model)
        result = self.data_model.get_dataframe()['new_value']
        self.assertEqual(0.0, result[0])
        self.assertEqual(2.0, result[1])
        self.assertEqual(105, round(result[2]))


if __name__ == '__main__':
    unittest.main()
