import unittest
from AIBuilder.Data import DataModel, MetaData
import pandas as pd
from datetime import datetime
from AIBuilder.DataScrubbing import MissingDataScrubber, StringToDateScrubber, AverageColumnScrubber, \
    ConvertCurrencyScrubber, AndScrubber, OutlierScrubber, MakeCategoricalScrubber, MultipleCatToListScrubber, \
    MultipleCatListToMultipleHotScrubber, BlackListScrubber


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


class TestBlackListScrubber(unittest.TestCase):

    def setUp(self):
        data = {
            'categorical': ['cat', 'dog', 'cat', 'cat', 'dog', 'bird', 'cat', 'cat', 'dog', 'bird', 'cat', 'dog'],
            'column': [12, 45, 23, 78, 4, 34, 1, 3, 89, 0, 1, 56],
        }

        metadata = MetaData()
        metadata.define_categorical_columns(['categorical'])
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def testValidateInvalid(self):
        scrubber = BlackListScrubber(column_name='invalid', black_list=['bird'])
        with self.assertRaises(AssertionError):
            scrubber.validate(self.data_model)

    def testValidateInvalid2(self):
        scrubber = BlackListScrubber(column_name='column', black_list=['bird'])
        with self.assertRaises(AssertionError):
            scrubber.validate(self.data_model)

    def testValidateValid(self):
        scrubber = BlackListScrubber(column_name='categorical', black_list=['cat', 'bird'])
        scrubber.validate(self.data_model)

    def testScrub(self):
        scrubber = BlackListScrubber(column_name='categorical', black_list=['bird'])
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()
        categories = df['categorical'].values.tolist()
        self.assertEqual(10, len(df))
        self.assertNotIn('bird', categories)

    def testScrub2(self):
        scrubber = BlackListScrubber(column_name='categorical', black_list=['dog', 'bird'])
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()
        categories = df['categorical'].values.tolist()
        self.assertEqual(6, len(df))
        self.assertNotIn('bird', categories)
        self.assertNotIn('dog', categories)


class TestOutlierScrubbing(unittest.TestCase):

    def setUp(self):
        data = {
            'num_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 50],
            'num_2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 50],
            'cat_1': ['EUR', 'USD', 'EUR', 'EUR', 'USD', 'EUR', 'EUR', 'USD', 'EUR', 'USD'],
        }

        metadata = MetaData()
        metadata.define_numerical_columns(['num_1', 'num_2'])
        metadata.define_categorical_columns(['cat_1'])
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def testOutlierScrubberCreate1(self):
        OutlierScrubber(col_z={'num_1': 2, 'num_2': 3})

    def testOutlierScrubberCreate2(self):
        OutlierScrubber(all_z=2)

    def testOutlierScrubberCreate3(self):
        with self.assertRaises(AssertionError):
            OutlierScrubber(col_z={'num_1': 2, 'num_2': 3}, all_z=2)

    def testOutlierScrubberCreate4(self):
        with self.assertRaises(AssertionError):
            OutlierScrubber()

    def testOutlierScrubberConfig(self):
        scrubber = OutlierScrubber(col_z={'num_1': 2, 'num_2': 3})
        config = scrubber.scrubber_config_list
        self.assertEqual(config['num_1'], MetaData.NUMERICAL_DATA_TYPE)
        self.assertEqual(config['num_2'], MetaData.NUMERICAL_DATA_TYPE)

    def testOutlierValidateMetaData(self):
        scrubber = OutlierScrubber(col_z={'num_1': 2, 'num_2': 3})
        scrubber.validate_metadata(self.data_model.metadata)

    def testOutlierValidateMetaDataFalse(self):
        scrubber = OutlierScrubber(col_z={'num_1': 2, 'num_3': 3})

        with self.assertRaises(RuntimeError):
            scrubber.validate_metadata(self.data_model.metadata)

    def testOutlierValidate(self):
        scrubber = OutlierScrubber(col_z={'num_1': 2, 'num_2': 3})
        scrubber.validate(self.data_model)

    def testOutlierValidateFalse(self):
        scrubber = OutlierScrubber(col_z={'num_1': 2, 'num_3': 3})

        with self.assertRaises(AssertionError):
            scrubber.validate(self.data_model)

    def setOutlierScrubber1(self):
        with self.assertRaises(AssertionError):
            outlier_scrubber = OutlierScrubber(col_z={'num_1': 2, 'cat_1': 3})
            outlier_scrubber.scrub(self.data_model)

    def testOutlierScrubber2(self):
        outlierScrubber = OutlierScrubber(col_z={'num_1': 2, 'num_2': 3})
        outlierScrubber.scrub(self.data_model)
        self.assertEqual(len(self.data_model.get_dataframe()), 9)

    def testOutlierScrubber3a(self):
        outlierScrubber = OutlierScrubber(all_z=2)
        outlierScrubber.scrub(self.data_model)
        self.assertEqual(len(self.data_model.get_dataframe()), 9)

    def testOutlierScrubber3b(self):
        outlierScrubber = OutlierScrubber(all_z=1)
        outlierScrubber.scrub(self.data_model)
        self.assertEqual(len(self.data_model.get_dataframe()), 5)


class TestMakeCategoricalScrubber(unittest.TestCase):

    def setUp(self):
        data = {
            'num_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 50],
            'cat_1': ['EUR', 'USD', 'EUR', 'EUR', 'USD', 'EUR', 'EUR', 'USD', 'EUR', 'USD'],
        }

        metadata = MetaData()
        metadata.define_numerical_columns(['num_1'])
        metadata.define_categorical_columns(['cat_1'])

        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def testScrubbing(self):
        scrubber = MakeCategoricalScrubber()
        scrubber.scrub(self.data_model)

        self.assertEqual(self.df['cat_1'].dtype, 'category')
        self.assertEqual(self.df['num_1'].dtype, 'int64')


class TestMultipleCatToListColumnScrubber(unittest.TestCase):

    def setUp(self):
        data = {
            'num_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 50],
            'mh_1': ['EUR,USD', 'USD,JPY,AUD', 'EUR', 'EUR,GBP,AUD', 'USD', 'EUR,JPY', 'EUR,GBP', 'USD,JPY', 'EUR,GBP',
                     'USD'],
        }

        metadata = MetaData()
        metadata.define_numerical_columns(['num_1'])
        metadata.define_multiple_cat_columns(['mh_1'])

        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def testScrubbing(self):
        scrubber = MultipleCatToListScrubber(sepperator=',')
        scrubber.scrub(self.data_model)

        first_item = self.data_model.get_dataframe()['mh_1'][0]
        self.assertIsInstance(first_item, list)


class TestMultipleCatListToMultipleHot(unittest.TestCase):

    def testScrubbing(self):
        data = {
            'num_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 50],
            'list_1': [['EUR', 'USD'], ['USD', 'JPY', 'AUD'], ['EUR'], ['EUR', 'GBP', 'AUD'], ['USD'], ['EUR', 'JPY'],
                       ['EUR', 'GBP'], ['USD', 'JPY'], ['EUR', 'GBP'],
                       ['USD']],
        }

        metadata = MetaData()
        metadata.define_numerical_columns(['num_1'])
        metadata.define_multiple_cat_columns(['list_1'])

        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

        scrubber = MultipleCatListToMultipleHotScrubber(col_name='list_1')
        scrubber.validate(self.data_model)
        scrubber.scrub(self.data_model)

        new_df = self.data_model.get_dataframe()
        columns = list(new_df.columns.values)

        # test new columns
        self.assertEqual(len(columns), 7)
        self.assertIn('list_1_EUR', columns)
        self.assertIn('list_1_GBP', columns)
        self.assertIn('list_1_USD', columns)
        self.assertIn('list_1_JPY', columns)
        self.assertIn('list_1_AUD', columns)

        # check column contents
        has_EUR_series = new_df['list_1_EUR']
        self.assertEqual(list(has_EUR_series.to_dict().values()), [1, 0, 1, 1, 0, 1, 1, 0, 1, 0])

        # test metadata
        meta_data_categorical_cols = self.data_model.metadata.binary_columns
        self.assertEqual(len(meta_data_categorical_cols), 5)
        self.assertIn('list_1_EUR', columns)
        self.assertIn('list_1_GBP', columns)
        self.assertIn('list_1_USD', columns)
        self.assertIn('list_1_JPY', columns)
        self.assertIn('list_1_AUD', columns)


if __name__ == '__main__':
    unittest.main()
