import numpy as np
import tensorflow as tf
import unittest
import pandas as pd
from AIBuilder import Data


def _dataset_to_dict(features):
    """ Convert a Pandas dataFrame to a dict with columns as key and values as arrays.

    :param features:
    :return:
    """

    return {key: np.array(value) for key, value in dict(features).items()}


def base_input_function(ml_dataset: Data.DataModel, batch_size=1, epoch=1):
    """ input function one, made for shoes AI.

    :param ml_dataset: Data.MLDataset
    :param epoch: int
    :param batch_size: int
    :return:
    """

    features = _dataset_to_dict(features=ml_dataset.get_feature_columns())

    print(ml_dataset)

    data_set = tf.data.Dataset.from_tensor_slices((features, ml_dataset.get_target_column()))

    data_set = data_set.shuffle(100).repeat(epoch).batch(batch_size)

    return data_set.make_one_shot_iterator().get_next()


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
        iterator = base_input_function(self._dataset)
        session = tf.Session()
        result1 = session.run(iterator)

        # compare to
        self.assertTrue(result1[1].item() == b'2')
        self.assertTrue(result1[0]['col2'].item() == b'4')
        self.assertTrue(result1[0]['col3'].item() == b'6')


if __name__ == '__main__':
    unittest.main()

