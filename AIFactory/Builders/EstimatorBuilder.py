import unittest
from unittest import mock
from AIBuilder.AI import AbstractAI, AI
from AIBuilder.AIFactory.Builders.Builder import Builder
import tensorflow as tf


class EstimatorBuilder(Builder):
    LINEAR_REGRESSOR = 'linear_regressor'
    valid_estimator_types = [LINEAR_REGRESSOR]
    estimator: str

    def __init__(self, estimator_type: str):
        self.estimator_type = None
        self.set_estimator(estimator_type)

    @property
    def dependent_on(self) -> list:
        return [self.OPTIMIZER, self.DATA_MODEL]

    @property
    def builder_type(self) -> str:
        return self.ESTIMATOR

    def set_estimator(self, estimator_type):
        self.validate_estimator(estimator_type)
        self.estimator_type = estimator_type

    def validate_estimator(self, estimator_type: str):
        assert estimator_type in self.valid_estimator_types, 'Unknown type of estimator {}, must be in {}'.format(
            estimator_type, self.valid_estimator_types)

    def validate(self) -> bool:
        self.validate_estimator(self.estimator_type)
        return True

    def build(self, neural_net: AbstractAI):
        if self.estimator_type is self.LINEAR_REGRESSOR:
            estimator = tf.estimator.LinearRegressor(
                feature_columns=neural_net.training_data.get_tf_feature_columns(),
                optimizer=neural_net.optimizer
            )

            neural_net.set_estimator(estimator)
            return

        raise RuntimeError('Estimator Builder failed to set estimator.')


class TestEstimatorBuilder(unittest.TestCase):

    def test_validate(self):
        estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)
        estimator_builder.validate()

    def test_invalid_estimator_type(self):
        invalid_estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)
        invalid_estimator_builder.estimator_type = 'invalid'

        valid_estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)

        with self.assertRaises(AssertionError):
            invalid_estimator_builder.validate()

        with self.assertRaises(AssertionError):
            valid_estimator_builder.set_estimator('invalid')

        with self.assertRaises(AssertionError):
            EstimatorBuilder('invalid')

    @mock.patch('AIFactory.tf.train.GradientDescentOptimizer')
    @mock.patch('AIFactory.DataModel')
    def test_build(self, mock_data_model, mock_optimizer):
        estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)
        arti = AI()

        mock_data_model.get_tf_feature_columns.return_value = []

        arti.set_optimizer(mock_optimizer)
        arti.set_training_data(mock_data_model)

        estimator_builder.build(arti)
        self.assertIsNotNone(arti.estimator)

