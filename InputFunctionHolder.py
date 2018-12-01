import numpy as np
import tensorflow as tf
from AIBuilder import Data


def _dataset_to_dict(features):
    """ Convert a Pandas dataFrame to a dict with columns as key and values as arrays.

    :param features:
    :return:
    """

    return {key: np.array(value) for key, value in dict(features).items()}


def base_fn(data_model: Data.DataModel, batch_size=1, epoch=1):
    """ input function one, made for shoes AI.

    :param data_model: Data.MLDataset
    :param epoch: int
    :param batch_size: int
    :return:
    """

    features = _dataset_to_dict(features=data_model.get_feature_columns())

    data_set = tf.data.Dataset.from_tensor_slices((features, data_model.get_target_column()))

    data_set = data_set.shuffle(100).repeat(epoch).batch(batch_size)

    return data_set.make_one_shot_iterator().get_next()
