import unittest
from AIBuilder.Data import DataModel
import pandas as pd
from datetime import datetime
from AIBuilder.DataScrubbing import MissingDataScrubber, StringToDateScrubber, AverageColumnScrubber, \
    ConvertCurrencyScrubber, AndScrubber


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
