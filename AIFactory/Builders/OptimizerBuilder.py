import unittest
from AIBuilder.AI import AbstractAI, AI
from AIBuilder.AIFactory.Builders.Builder import Builder
import tensorflow as tf


class OptimizerBuilder(Builder):
    LEARNING_RATE = 'learning_rate'
    GRADIENT_CLIPPING = 'gradient_clipping'

    GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent_optimizer'

    valid_optimizer_types = [GRADIENT_DESCENT_OPTIMIZER]

    def __init__(self, optimizer_type: str, learning_rate: float, gradient_clipping: Optional[float] = None):
        self.validate_optimizer_type(optimizer_type)
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def ingredient_type(self) -> str:
        return self.OPTIMIZER

    def validate(self) -> bool:
        assert self.learning_rate is not float, 'optimizer learning rate must be float, currently: {}'.format(
            self.learning_rate)

        self.validate_optimizer_type(self.optimizer_type)

        assert type(self.gradient_clipping) is float or self.gradient_clipping is None, \
            'gradient clipping must be float or None, currently {}'.format(self.gradient_clipping)

    def build(self, neural_net: AbstractAI):
        my_optimizer = self._set_optimizer(optimizer_type=self.optimizer_type, learning_rate=self.learning_rate)

        if self.gradient_clipping is not None:
            my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, self.gradient_clipping)

        neural_net.set_optimizer(my_optimizer)

    def _set_optimizer(self, optimizer_type: str, learning_rate: float) -> tf.train.Optimizer:
        if optimizer_type is self.GRADIENT_DESCENT_OPTIMIZER:
            my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            return my_optimizer

        raise RuntimeError('Optimizer not set.')

    def validate_optimizer_type(self, optimizer_type: str):
        assert optimizer_type in self.valid_optimizer_types, 'Unknown type op optimizer {}, must be in {}'.format(
            optimizer_type, self.valid_optimizer_types)


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
        with self.assertRaises(AssertionError):
            OptimizerBuilder(
                optimizer_type='invalid',
                learning_rate=5.0,
                gradient_clipping=0.0002)

        optimizer_builder = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=0.1,
            gradient_clipping=0.0002)

        optimizer_builder.optimizer_type = 'invalid'

        with self.assertRaises(AssertionError):
            optimizer_builder.validate()

    def test_build_with_clipping(self):
        arti = AI()
        optimizer_builder_with_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0,
            gradient_clipping=1.0)

        optimizer_builder_with_clipping.build(arti)
        self.assertIsNotNone(arti.optimizer)

    def test_build_with_no_clipping(self):
        arti = AI()
        optimizer_builder_no_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0)

        optimizer_builder_no_clipping.build(arti)
        self.assertIsNotNone(arti.optimizer)
