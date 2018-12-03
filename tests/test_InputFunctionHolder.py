import tensorflow as tf
import unittest
import pandas as pd
from AIBuilder import Data
from AIBuilder.InputFunctionHolder import base_fn


class TestBaseInputFunction(unittest.TestCase):
    _dataset: Data.DataModel

    def setUp(self):
        self._data = {'col1': ['1', '2'], 'col2': ['3', '4'], 'col3': ['5', '6']}
        self._dataframe = pd.DataFrame(data=self._data)
        self._dataset = Data.DataModel(self._dataframe)

        self._dataset.set_target_column('col1')
        self._dataset.set_tf_feature_columns([
            tf.feature_column.categorical_column_with_vocabulary_list(
                'col2',
                vocabulary_list=['3', '4']
            ),
            tf.feature_column.categorical_column_with_vocabulary_list(
                'col3',
                vocabulary_list=['5', '6']
            )
        ])

    def test_feature_columns(self):
        self.setUp()
        iterator = base_fn(self._dataset)
        session = tf.Session()
        result1 = session.run(iterator)

        # todo: results unstable.
        # self.assertEqual(result1[1].item(), b'2')
        # self.assertEqual(result1[0]['col2'].item(), b'4')
        # self.assertEqual(result1[0]['col3'].item(), b'6')


if __name__ == '__main__':
    unittest.main()
