import unittest

from AIBuilder.AIFactory import BalanceData
from AIBuilder.Data import DataModel, MetaData
import pandas as pd
import numpy as np
from datetime import datetime
from AIBuilder.DataScrubbing import MissingDataReplacer, StringToDateScrubber, AverageColumnScrubber, \
    ConvertCurrencyScrubber, AndScrubber, OutlierScrubber, MakeCategoricalScrubber, MultipleCatToListScrubber, \
    MultipleCatListToMultipleHotScrubber, BlacklistCatScrubber, ConvertToColumnScrubber, CategoryToFloatScrubber, \
    CategoryByKeywordsFinder, MissingDataScrubber, ConvertToNumericScrubber, BinaryResampler, UnbalancedDataStrategy, \
    BlacklistTokenScrubber, CodeScrubber, PunctuationScrubber, StopWordScrubber, LowerTextScrubber, \
    WordStemmer


class TestConvertToNumericScrubber(unittest.TestCase):

    def setUp(self):
        self.data = {
            'col_1': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'col_2': ['0.2', '1.2', '2.2', '3.2', '4.2', '5.2', '6.2', '7.2', '8.2', '9.2'],
        }

        self.data_frame = pd.DataFrame(self.data)
        self.data_model = DataModel(self.data_frame)
        self.data_model.metadata.define_numerical_columns(['col_1', 'col_2'])

    def testScrub(self):
        scrubber = ConvertToNumericScrubber(column_names=['col_1', 'col_2'])
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()
        for value in df['col_1'].values:
            self.assertTrue(type(value) == np.int64)

        for value in df['col_2'].values:
            self.assertTrue(type(value) == np.float64)


class TestMissingDataScrubber(unittest.TestCase):

    def setUp(self):
        self.data = {
            'col_1': [0, 1, None, 3, 4, 5, 6, 7, 8, 9],
            'col_2': [0, 1, 2, 3, 4, 5, 6, 7, None, 9],
            'col_3': [np.nan, 'b', 'c', 'd', 'e', 'f', 'g', None, 'i', 'j'],
        }

        self.data_frame = pd.DataFrame(self.data)
        self.data_model = DataModel(self.data_frame)
        self.data_model.metadata.define_numerical_columns(['col_1', 'col_2'])
        self.data_model.metadata.define_categorical_columns(['col_3'])

    def test_scrubbing(self):
        scrubber = MissingDataScrubber(scrub_columns=['col_1', 'col_2', 'col_3'])
        result = scrubber.scrub(self.data_model)
        self.assertEqual(len(result), 6)
        for value in result.get_dataframe().values:
            self.assertFalse(np.isnan(value[0]))
            self.assertFalse(np.isnan(value[1]))


class TestMissingDataReplacer(unittest.TestCase):

    def setUp(self):
        self._data = {
            'numerical_1': [2, None, 3, 4],
            'numerical_3': [None, 2, 3, 4],
            'categorical_1': ['one', None, 'two', 'three'],
            'categorical_2': ['apple', 'pie', None, 'three'],
            'categorical_3': ['apple', 'pie', None, 'three'],
            'unknown_1': [9, 10, 11, 12]
        }

        self._dataframe = pd.DataFrame(data=self._data)
        self._data_model = DataModel(self._dataframe)

    def test_categorize_columns(self):
        categorical = ['categorical_1', 'categorical_2', 'categorical_3']
        self._data_model.metadata.define_categorical_columns(categorical)

        numerical = ['numerical_1', 'numerical_3']
        self._data_model.metadata.define_numerical_columns(numerical)

        unknown_category = 'unknown'
        missing_data_scrubber1 = MissingDataReplacer(missing_category_name=unknown_category,
                                                     missing_numerical_value='average',
                                                     scrub_columns=['categorical_1', 'categorical_2', 'numerical_1'])
        missing_data_scrubber1.validate(self._data_model)
        missing_data_scrubber1.scrub(self._data_model)

        missing_data_scrubber2 = MissingDataReplacer(missing_numerical_value=1, scrub_columns=['numerical_3'])
        missing_data_scrubber2.validate(self._data_model)
        missing_data_scrubber2.scrub(self._data_model)

        self.assertEqual(unknown_category, self._data_model.get_dataframe()['categorical_1'][1])
        self.assertEqual(unknown_category, self._data_model.get_dataframe()['categorical_2'][2])
        self.assertEqual(None, self._data_model.get_dataframe()['categorical_3'][2])
        self.assertEqual(2, self._data_model.get_dataframe()['numerical_1'][0])
        self.assertEqual(3, self._data_model.get_dataframe()['numerical_1'][1])
        self.assertEqual(1, self._data_model.get_dataframe()['numerical_3'][0])
        self.assertEqual(2, self._data_model.get_dataframe()['numerical_3'][1])
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


class TestBlacklistCatScrubber(unittest.TestCase):

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
        scrubber = BlacklistCatScrubber(column_name='invalid', blacklist=['bird'])
        with self.assertRaises(AssertionError):
            scrubber.validate(self.data_model)

    def testValidateInvalid2(self):
        scrubber = BlacklistCatScrubber(column_name='column', blacklist=['bird'])
        with self.assertRaises(AssertionError):
            scrubber.validate(self.data_model)

    def testValidateValid(self):
        scrubber = BlacklistCatScrubber(column_name='categorical', blacklist=['cat', 'bird'])
        scrubber.validate(self.data_model)

    def testScrub(self):
        scrubber = BlacklistCatScrubber(column_name='categorical', blacklist=['bird'])
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()
        categories = df['categorical'].values.tolist()
        self.assertEqual(10, len(df))
        self.assertNotIn('bird', categories)

    def testScrub2(self):
        scrubber = BlacklistCatScrubber(column_name='categorical', blacklist=['dog', 'bird'])
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()
        categories = df['categorical'].values.tolist()
        self.assertEqual(6, len(df))
        self.assertNotIn('bird', categories)
        self.assertNotIn('dog', categories)

    def testScrub3(self):
        scrubber = BlacklistCatScrubber(column_name='categorical', blacklist=['python'])
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()
        categories = df['categorical'].values.tolist()
        self.assertEqual(12, len(df))


class TestBlacklistTokenScrubber(unittest.TestCase):

    def setUp(self):
        data = {
            'target_column': [
                ['Job', 'in', 'Dixon', 'with', 'successful', 'business'],
                ['Engineer', 'Quality', 'in', 'Dixon'],
                ['Shift', 'Supervisor', 'Part', 'time', 'job', 'in', 'Camphill'],
                ['Construction', 'PM', 'Job', 'in', 'Dixon'],
                ['CyberCoders', 'Application', 'Principal', 'QA', 'Engineer', 'Java'],
            ],
            'column': [12, 45, 23, 78, 4],
        }

        metadata = MetaData()
        metadata.define_list_columns(['target_column'])
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata
        self.blacklist = ['job', 'in', 'Dixon']

    def testValidateInvalid(self):
        scrubber = BlacklistTokenScrubber(column_name='invalid', blacklist=self.blacklist)
        with self.assertRaises(AssertionError):
            scrubber.validate(self.data_model)

    def testValidateInvalid2(self):
        scrubber = BlacklistTokenScrubber(column_name='column', blacklist=self.blacklist)
        with self.assertRaises(AssertionError):
            scrubber.validate(self.data_model)

    def testValidateValid(self):
        scrubber = BlacklistTokenScrubber(column_name='target_column', blacklist=self.blacklist)
        scrubber.validate(self.data_model)

    def testScrub(self):
        scrubber = BlacklistTokenScrubber(column_name='target_column', blacklist=self.blacklist)
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()
        lists = df['target_column'].values.tolist()
        self.assertEqual(5, len(df))
        for row in lists:
            for item in self.blacklist:
                self.assertNotIn(item, row)

    def testScrubWithSynonyms(self):
        scrubber = BlacklistTokenScrubber(column_name='target_column', blacklist=self.blacklist, use_synonyms=True)
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()
        lists = df['target_column'].values.tolist()
        self.assertEqual(5, len(df))
        self.blacklist.append('business')
        for row in lists:
            for item in self.blacklist:
                self.assertNotIn(item, row)


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


class TestConvertToColumn(unittest.TestCase):

    def setUp(self):
        data = {
            'num_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 50],
            'num_2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        metadata = MetaData()
        metadata.define_numerical_columns(['num_1', 'num_2'])
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def testScrubbing(self):
        def convert(row):
            return row['num_1'] + row['num_2']

        scrubber = ConvertToColumnScrubber(new_column_name='num_3', converter=convert,
                                           new_column_type=MetaData.NUMERICAL_DATA_TYPE,
                                           required_columns={'num_1': MetaData.NUMERICAL_DATA_TYPE,
                                                             'num_2': MetaData.NUMERICAL_DATA_TYPE})
        scrubber.validate(self.data_model)
        result = scrubber.scrub(self.data_model)
        result_df = result.get_dataframe()

        for row in result_df.values:
            self.assertEqual(row[2], row[0] + 1)

    def testInvalid(self):
        def convert(row):
            return row['num_1'] + row['num_2']

        scrubber = ConvertToColumnScrubber(new_column_name='num_3', converter=convert,
                                           new_column_type=MetaData.NUMERICAL_DATA_TYPE,
                                           required_columns={'num_1': MetaData.NUMERICAL_DATA_TYPE,
                                                             'num_4': MetaData.NUMERICAL_DATA_TYPE})

        with self.assertRaises(AssertionError):
            scrubber.validate(self.data_model)


class TestCategoryToFloatScrubber(unittest.TestCase):
    def setUp(self):
        data = {
            'cat_1': ['one', 'two', 'three', 'two', 'one'],
        }

        metadata = MetaData()
        metadata.define_categorical_columns(['cat_1'])

        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def testScrubbing(self):
        cat_to_value_index = {'one': 1, 'two': 2, 'three': 3}
        scrubber = CategoryToFloatScrubber(new_column_name='num_1',
                                           source_column_name='cat_1',
                                           category_to_value_index=cat_to_value_index)

        scrubber.validate(self.data_model)
        result = scrubber.scrub(self.data_model)
        result_df = result.get_dataframe()

        for result in result_df.values:
            self.assertEqual(result[1], cat_to_value_index[result[0]])


class TestKeyWordToCategoryScrubber(unittest.TestCase):
    def setUp(self):
        data = {
            'text_1': ['i am one', 'i am I', 'we are two', 'the three of us', 'two become one', 'we are legion',
                       '2 become 1', 'we are foo bar'],
        }

        metadata = MetaData()
        metadata.define_text_columns(['text_1'])

        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def testScrubbing(self):
        scrubber = CategoryByKeywordsFinder(new_column_name='cat_1',
                                            source_column_name='text_1',
                                            category_keywords_map={'one': ['one'], 'two': ['two'], 'three': ['three']},
                                            unknown_category='unknown',
                                            verbosity=0)

        scrubber.validate(self.data_model)
        result = scrubber.scrub(self.data_model)
        result_df = result.get_dataframe()

        self.assertEqual(result_df['cat_1'].values.tolist(),
                         ['one', 'unknown', 'two', 'three', 'one', 'unknown', 'unknown', 'unknown'])

    def testScrubbingWithVoting(self):
        scrubber = CategoryByKeywordsFinder(new_column_name='cat_1', source_column_name='text_1',
                                            category_keywords_map={'a': ['we', 'i'],
                                                                   'b': ['two', 'three', 'one', 'foo', 'bar']},
                                            unknown_category='unknown', verbosity=0)

        scrubber.validate(self.data_model)
        result = scrubber.scrub(self.data_model)
        result_df = result.get_dataframe()

        self.assertEqual(['a', 'a', 'a', 'b', 'b', 'a', 'unknown', 'b'],
                         result_df['cat_1'].values.tolist())

    def testScrubbingWithMultipleCat(self):
        scrubber = CategoryByKeywordsFinder(new_column_name='cat_1', source_column_name='text_1',
                                            category_keywords_map={'a': ['we', 'i'],
                                                                   'b': ['two', 'three', 'one', 'foo', 'bar']},
                                            unknown_category='unknown', verbosity=0, multiple_cats=True)

        scrubber.validate(self.data_model)
        result = scrubber.scrub(self.data_model)
        result_df = result.get_dataframe()

        self.assertEqual([{'a', 'b'}, {'a'}, {'a', 'b'}, {'b'}, {'b'}, {'a'}, {'unknown'}, {'a', 'b'}],
                         result_df['cat_1'].values.tolist())

    def testScrubbingWithAliases(self):
        scrubber = CategoryByKeywordsFinder(new_column_name='cat_1',
                                            source_column_name='text_1',
                                            category_keywords_map={'one': ['one'], 'two': ['two'], 'three': ['three'],
                                                                   'foo bar': ['foo bar']},
                                            unknown_category='unknown',
                                            use_synonyms=True,
                                            min_syntactic_distance=0.01,
                                            verbosity=0)

        scrubber.validate(self.data_model)
        result = scrubber.scrub(self.data_model)
        result_df = result.get_dataframe()

        self.assertEqual(result_df['cat_1'].values.tolist(),
                         ['one', 'one', 'two', 'three', 'one', 'unknown', 'one', 'foo bar'])


class TestBinaryResampler(unittest.TestCase):

    def setUp(self):
        data = {
            'num_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 50],
            'cat_2': [1, 0, 1, 0, 1, 1, 1, 1, 0, 1]
        }

        metadata = MetaData()
        metadata.define_numerical_columns(['num_1'])
        metadata.define_categorical_columns(['cat_2'])
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def test_validate(self):
        scrubber = BinaryResampler('cat_2', UnbalancedDataStrategy.UNDER_SAMPLING)
        scrubber.validate_metadata(self.data_model.metadata)
        scrubber.validate(self.data_model)

    def test_scrub_under_sampling(self):
        scrubber = BinaryResampler('cat_2', UnbalancedDataStrategy.UNDER_SAMPLING)
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()

        self.assertEqual(len(df), 6)
        self.assertEqual(0.5, df['cat_2'].mean())

    def test_scrub_over_sampling(self):
        scrubber = BinaryResampler('cat_2', UnbalancedDataStrategy.OVER_SAMPLING)
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()

        self.assertEqual(len(df), 14)
        self.assertEqual(0.5, df['cat_2'].mean())

    def test_re_weigh(self):
        scrubber = BinaryResampler('cat_2', UnbalancedDataStrategy.RE_WEIGH)
        result = scrubber.scrub(self.data_model)
        df = result.get_dataframe()

        weights_for_0 = []
        weights_for_1 = []
        for row in df.values:
            if row[1] == 1:
                weights_for_1.append(row[2])
            elif row[1] == 0:
                weights_for_0.append(row[2])

        # check all row with the same target category have got the same weight.
        self.assertEqual(1, len(set(weights_for_0)))
        self.assertEqual(1, len(set(weights_for_1)))

        # all weights combined should approx. equal the length of the dataframe because that indicate an average weight of 1.
        self.assertAlmostEqual(len(df), sum(df[BalanceData.WEIGHTS_COLUMN].values))


# class testDataRegrouper(unittest.TestCase):
#
#     def setUp(self) -> None:
#         n = 12
#         data = {
#             'attr_one': 6 * ['ham'] + 1 * ['spam'] + 5 * ['eggs'],
#             'attr_two': int(n / 2) * [1, 0],
#             'text': n * ['this is some text ']
#         }
#
#         metadata = MetaData()
#         self.df = pd.DataFrame(data)
#         self.data_model = DataModel(self.df)
#         self.data_model.metadata = metadata
#
#     def testRegroupingWithSize2(self):
#         scrubber = DataRowMerger(group_by='attr_two', spread_by='attr_one', group_size=2)
#         scrubber.scrub(self.data_model)
#         df = self.data_model.get_dataframe()
#
#         for merged_title in df['attr_one']:
#             self.assertLessEqual(merged_title.count('ham'), 1)
#             self.assertLessEqual(merged_title.count('spam'), 1)
#             self.assertLessEqual(merged_title.count('eggs'), 1)


class testHTMLScrubber(unittest.TestCase):

    def setUp(self):
        data = {'text': ['<p>this is some text</p>',
                         '<h1 id="bla", class="blabla", style="transform: translatyeY(-50%)">this is some more text</h1>',
                         '<p>and even <b>more</b> text, damn</p>',
                         'this is my text (dont remove this)',
                         "this is some text,  $('span#TrackingJobBody a').each(function (i, v) { if ($(v).attr('href')) { var href = $(v).attr('href').toLowerCase(); if (href.match(\"^http\")) { switch (true) { case /facebook/.test(href): $(v).attr('mns_rt', 'NonJob-Facebook'); break; case /linkedin/.test(href): $(v).attr('mns_rt', 'NonJob-Linkedin'); break; case /twitter\.com/.test(href): $(v).attr('mns_rt', 'NonJob-Twitter'); break; case /plus\.google\.com/.test(href): $(v).attr('mns_rt', 'NonJob-GooglePlus'); break; case /youtube/.test(href): $(v).attr('data-track', 'Client-Social-Youtube'); break; case /http[s]?\:\/\/([a-z0-9\-\.]{1,}\.[a-z]{2,})[\/]?$/.test(href): $(v).attr('data-track', 'Client-Link-Homepage'); break; default: $(v).attr('mns_rt', 'jobcustomapplyonline'); break; } } } });"
                         ]
                }

        metadata = MetaData()
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def testHTMLRemoval(self):
        scrubber = CodeScrubber('text')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for text in df['text']:
            self.validate_string(text)

    def validate_string(self, text):
        print(text)
        self.assertFalse('>' in text)
        self.assertFalse('<' in text)
        self.assertFalse('{' in text)
        self.assertFalse('bla' in text)
        self.assertFalse('transform' in text)
        self.assertFalse('50%' in text)
        self.assertTrue('dont remove this' in text)

    def testHTMLRemoval_verbose(self):
        scrubber = CodeScrubber('text', verbosity=1)
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for text in df['text']:
            self.validate_string(text)

    def testHTMLRemoval_new_col(self):
        scrubber = CodeScrubber(text_column='text', new_text_column='new')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for text in df['new']:
            self.validate_string(text)


class testPunctuationScrubber(unittest.TestCase):

    def setUp(self):
        data = {'text': ['This is, some text.', 'Is this, (some) text!?',
                         'Would you like: ham, spam and eggs; spam, ham and eggs or eggs, ham and spam?']}

        metadata = MetaData()
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def test_punc_removal(self):
        scrubber = PunctuationScrubber('text')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for text in df['text']:
            self.validate_string(text)

    def validate_string(self, text):
        self.assertFalse(',' in text)
        self.assertFalse('.' in text)
        self.assertFalse(';' in text)
        self.assertFalse(':' in text)
        self.assertFalse('!' in text)
        self.assertFalse('?' in text)
        self.assertFalse('(' in text)
        self.assertFalse(')' in text)

    def test_punc_removal_verbose(self):
        scrubber = PunctuationScrubber('text', verbosity=1)
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for text in df['text']:
            self.validate_string(text)

    def test_punc_removal_new_col(self):
        scrubber = PunctuationScrubber(text_column='text', new_text_column='new')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for text in df['new']:
            self.validate_string(text)


class testLowerTextScrubber(unittest.TestCase):

    def setUp(self):
        data = {'text': ['This is some text.', 'Get some text ASAP?', 'This is some text for John and Joan']}

        metadata = MetaData()
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def test_lower(self):
        scrubber = LowerTextScrubber('text')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for text in df['text']:
            self.validate_string(text)

    def validate_string(self, text: str):
        self.assertEqual(text, text.lower())

    def test_lower_verbose(self):
        scrubber = LowerTextScrubber('text', verbosity=1)
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for text in df['text']:
            self.validate_string(text)

    def test_lower_new_col(self):
        scrubber = LowerTextScrubber(text_column='text', new_text_column='new')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for text in df['new']:
            self.validate_string(text)


class StopWordScrubberTest(unittest.TestCase):

    def setUp(self):
        data = {'test': [['this', 'sentence', 'has', 'multiple', 'stopwords'],
                         ['this', 'sentence', 'one', 'multiple', 'too'], ['verb', 'noun'], ['too', 'than', 'can']]
                }

        metadata = MetaData()
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def test_scrubbing_new_column(self):
        scrubber = StopWordScrubber(column='test', new_column='test2')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for row in df['test2']:
            self.assertFalse('this' in row)
            self.assertFalse('has' in row)
            self.assertFalse('too' in row)
            self.assertFalse('than' in row)
            self.assertFalse('can' in row)

        self.assertEqual(5, len(df['test'][0]))
        self.assertEqual(5, len(df['test'][1]))
        self.assertEqual(2, len(df['test'][2]))
        self.assertEqual(3, len(df['test'][3]))

    def test_scrubbing(self):
        scrubber = StopWordScrubber(column='test')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for row in df['test']:
            self.assertFalse('this' in row)
            self.assertFalse('has' in row)
            self.assertFalse('too' in row)
            self.assertFalse('than' in row)
            self.assertFalse('can' in row)

    def test_verbose_scrubbing(self):
        scrubber = StopWordScrubber(column='test', verbosity=1)
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        for row in df['test']:
            self.assertFalse('this' in row)
            self.assertFalse('has' in row)
            self.assertFalse('too' in row)
            self.assertFalse('than' in row)
            self.assertFalse('can' in row)


class WordStemmerTest(unittest.TestCase):

    def setUp(self):
        data = {'test': [['is', 'this', 'a', 'stemmable', 'sentence'],
                         ['cats', 'are', 'smarter', 'than', 'dogs']]
                }

        metadata = MetaData()
        self.df = pd.DataFrame(data)
        self.data_model = DataModel(self.df)
        self.data_model.metadata = metadata

    def test_scrubbing(self):
        scrubber = WordStemmer(column='test')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        self.assertEqual(df['test'][0], ['is', 'this', 'a', 'stemmabl', 'sentenc'])
        self.assertEqual(df['test'][1], ['cat', 'are', 'smarter', 'than', 'dog'])

    def test_scrubbing_verbose(self):
        scrubber = WordStemmer(column='test', verbosity=1)
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        self.assertEqual(df['test'][0], ['is', 'this', 'a', 'stemmabl', 'sentenc'])
        self.assertEqual(df['test'][1], ['cat', 'are', 'smarter', 'than', 'dog'])

    def test_scrubbing_new_col(self):
        scrubber = WordStemmer(column='test', new_column='new')
        scrubber.scrub(self.data_model)

        df = self.data_model.get_dataframe()
        self.assertEqual(df['new'][0], ['is', 'this', 'a', 'stemmabl', 'sentenc'])
        self.assertEqual(df['new'][1], ['cat', 'are', 'smarter', 'than', 'dog'])
        self.assertEqual(df['test'][0], ['is', 'this', 'a', 'stemmable', 'sentence'])
        self.assertEqual(df['test'][1], ['cats', 'are', 'smarter', 'than', 'dogs'])


if __name__ == '__main__':
    unittest.main()
