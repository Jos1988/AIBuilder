import unittest
from unittest import mock
from AIBuilder.AIFactory.Specifications.BasicSpecifications import TypeSpecification
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Builders.Builder import Builder
import tensorflow as tf


class EstimatorBuilder(Builder):
    LINEAR_REGRESSOR = 'linear_regressor'
    valid_estimator_types = [LINEAR_REGRESSOR]
    estimator: str
    estimator_type = None

    def __init__(self, estimator_type: str):
        self.set_estimator(estimator_type)

    @property
    def dependent_on(self) -> list:
        return [self.OPTIMIZER, self.DATA_MODEL]

    @property
    def builder_type(self) -> str:
        return self.ESTIMATOR

    def set_estimator(self, estimator_type):
        self.estimator_type = TypeSpecification('estimator_type', estimator_type, self.valid_estimator_types)

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        if self.estimator_type() is self.LINEAR_REGRESSOR:
            estimator = tf.estimator.LinearRegressor(
                feature_columns=neural_net.training_data.get_tf_feature_columns(),
                optimizer=neural_net.optimizer,
                model_dir=self.render_model_dir(neural_net)
            )

            neural_net.set_estimator(estimator)
            return

        raise RuntimeError('Estimator Builder failed to set estimator.')

    @staticmethod
    def render_model_dir(ai) -> str:
        return ai.get_log_dir() + '/' + ai.get_project_name() + '/tensor_board/' + ai.get_name()


class TestEstimatorBuilder(unittest.TestCase):

    def test_validate(self):
        estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)
        estimator_builder.validate()

    def test_invalid_estimator_type(self):
        invalid_estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)
        invalid_estimator_builder.estimator_type = TypeSpecification(name=EstimatorBuilder.ESTIMATOR,
                                                                     value='invalid',
                                                                     valid_types=EstimatorBuilder.valid_estimator_types)

        valid_estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)

        with self.assertRaises(AssertionError):
            invalid_estimator_builder.validate()

        with self.assertRaises(AssertionError):
            valid_estimator_builder.set_estimator('invalid')
            valid_estimator_builder.validate()

        with self.assertRaises(AssertionError):
            builder = EstimatorBuilder('invalid')
            builder.validate()

    def test_build(self):
        mock_data_model = mock.MagicMock()
        mock_optimizer = mock.MagicMock()

        estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)

        mock_data_model.get_tf_feature_columns.return_value = []

        arti = mock.Mock('EstimatorBuilder.AbstractAI')
        arti.set_estimator = mock.Mock()
        arti.optimizer = mock_optimizer
        arti.training_data = mock_data_model

        estimator_builder.build(arti)
        arti.set_estimator.assert_called_once()


if __name__ == '__main__':
    unittest.main()
