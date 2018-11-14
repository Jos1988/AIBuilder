import unittest
from unittest import mock
from typing import Optional
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Builders.Builder import Builder
from AIBuilder.AIFactory.Specifications.BasicSpecifications import DataTypeSpecification, TypeSpecification, \
    NullSpecification
import tensorflow as tf


class OptimizerBuilder(Builder):
    LEARNING_RATE = 'learning_rate'
    GRADIENT_CLIPPING = 'gradient_clipping'

    GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent_optimizer'

    valid_optimizer_types = [GRADIENT_DESCENT_OPTIMIZER]

    def __init__(self, optimizer_type: str, learning_rate: float, gradient_clipping: Optional[float] = None):
        self.optimizer_type = TypeSpecification('optimizer_type', optimizer_type, self.valid_optimizer_types)
        self.learning_rate = DataTypeSpecification('learning_rate', learning_rate, float)

        self.gradient_clipping = NullSpecification('gradient_clipping')
        if gradient_clipping is not None:
            self.gradient_clipping = DataTypeSpecification('gradient_clipping', gradient_clipping, float)

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def builder_type(self) -> str:
        return self.OPTIMIZER

    def validate(self):
        self.validate_specifications()

    def build(self, neural_net: AbstractAI):
        my_optimizer = self._set_optimizer(optimizer_type=self.optimizer_type(), learning_rate=self.learning_rate())

        if self.gradient_clipping is not None:
            my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, self.gradient_clipping)

        neural_net.set_optimizer(my_optimizer)

    def _set_optimizer(self, optimizer_type: str, learning_rate: float) -> tf.train.Optimizer:
        if optimizer_type is self.GRADIENT_DESCENT_OPTIMIZER:
            my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            return my_optimizer

        raise RuntimeError('Optimizer not set.')


class TestOptimizerBuilder(unittest.TestCase):

    def test_valid_validate(self):
        optimizer_builder_no_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0)

        optimizer_builder_with_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0,
            gradient_clipping=1.0)

        optimizer_builder_no_clipping.validate()
        optimizer_builder_with_clipping.validate()

    def test_invalid_validate(self):
        optimizer_builder = OptimizerBuilder(
            optimizer_type='invalid',
            learning_rate=5.0,
            gradient_clipping=0.0002)

        with self.assertRaises(AssertionError):
            optimizer_builder.validate()

    @mock.patch('OptimizerBuilder.tf.contrib.estimator.clip_gradients_by_norm')
    @mock.patch('OptimizerBuilder.tf.train.GradientDescentOptimizer')
    def test_build_with_clipping(self, mock_optimizer, mock_clipper):
        arti = mock.Mock('OptimizerBuilder.AbstractAI')
        arti.set_optimizer = mock.Mock()

        optimizer_builder_with_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0,
            gradient_clipping=0.1)

        optimizer_builder_with_clipping.build(arti)
        arti.set_optimizer.assert_called_once()
        mock_optimizer.assert_called_with(learning_rate=1.0)
        mock_clipper.assert_called()

    @mock.patch('OptimizerBuilder.tf.train.GradientDescentOptimizer')
    def test_build_with_no_clipping(self, mock_optimizer):
        arti = mock.Mock('OptimizerBuilder.AbstractAI')
        arti.set_optimizer = mock.Mock()

        optimizer_builder_no_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0)

        optimizer_builder_no_clipping.build(arti)
        arti.set_optimizer.assert_called_once()
        mock_optimizer.assert_called_with(learning_rate=1.0)


if __name__ == '__main__':
    unittest.main()
