import pandas as pd
import unittest
import tensorflow as tf
from AIBuilder.Data import MetaData, DataModel, DataSetSplitter


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

    def test_unknown_is_default(self):
        self.assertEqual(self.meta_data.get_column_type('categorical_2'), 'unknown')

    def test_to_string(self):
        self.meta_data.define_categorical_columns(['categorical_1', 'categorical_2', 'categorical_3'])
        self.meta_data.define_numerical_columns(['numerical_1', 'numerical_2', 'numerical_3'])

        expected_string = '\nmetadata:\ncategorical_1: ' + MetaData.CATEGORICAL_DATA_TYPE + '\ncategorical_2: ' + MetaData.CATEGORICAL_DATA_TYPE + '\ncategorical_3: ' + MetaData.CATEGORICAL_DATA_TYPE + '\nnumerical_1: ' + MetaData.NUMERICAL_DATA_TYPE + '\nnumerical_2: ' + MetaData.NUMERICAL_DATA_TYPE + '\nnumerical_3: ' + MetaData.NUMERICAL_DATA_TYPE + '\nunknown_1: ' + MetaData.UNKNOWN_DATA_TYPE
        stringified_metadata = str(self.meta_data)
        self.assertEqual(expected_string, stringified_metadata, 'Metadata __str__ function does not meet expectations.')


class TestDataset(unittest.TestCase):

    def setUp(self):
        self._data = {'col1': [1, 2], 'col2': [3, 4], 'col4': [5, 6]}
        self._dataframe = pd.DataFrame(data=self._data)
        self._dataset = DataModel(self._dataframe)

    def test_validate_columns_invalid(self):
        with self.assertRaises(RuntimeError):
            self._dataset.validate_columns(['col3'])

    def test_validate_columns(self):
        self._dataset.validate_columns(['col1'])

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


if __name__ == '__main__':
    unittest.main()
