from abc import ABC, abstractmethod
from typing import List, Optional
import tensorflow as tf
from AIBuilder import AI


class OptimizerStrategy(ABC):
    GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent_optimizer'
    ADAM_OPTIMIZER = 'adam_optimizer'
    ADAGRAD_OPTIMIZER = 'adagrad_optimizer'
    ADADELTA_OPTIMIZER = 'adadelta_optimizer'

    def __init__(self, ml_model: AI, learning_rate: int, gradient_clipping: Optional[int] = None, **kwargs):
        self.gradient_clipping = None
        if gradient_clipping is not None:
            self.gradient_clipping = gradient_clipping

        self.learning_rate = learning_rate
        self.ML_Model = ml_model
        self.kwargs = kwargs
        self.result = None

    def build(self):
        self.result = self.build_optimizer()
        self.validate_result()

        return self.result

    def set_optimizer_kwargs(self) -> dict:
        kwargs = {
            'learning_rate': self.learning_rate
        }
        kwargs.update(self.kwargs)

        return kwargs

    def set_gradient_clipping(self, optimizer):
        if self.gradient_clipping is not None:
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, self.gradient_clipping)
        return optimizer

    @abstractmethod
    def build_optimizer(self):
        pass

    @abstractmethod
    def validate_result(self):
        pass

    @staticmethod
    @abstractmethod
    def optimizer_type() -> str:
        pass


class GradientDescentOptimizerStrategy(OptimizerStrategy):
    """ Basic Gradient descent optimization.
    """

    def build_optimizer(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        optimizer = self.set_gradient_clipping(optimizer)

        return optimizer

    def validate_result(self):
        if self.gradient_clipping is None:
            assert isinstance(self.result, tf.train.GradientDescentOptimizer)

    @staticmethod
    def optimizer_type() -> str:
        return OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER


class AdagradOptimizerStrategy(OptimizerStrategy):
    """ Adagrad optimizer, Adaptive gradient learning rates, but they always decay.
    """
    def build_optimizer(self):
        kwargs = self.set_optimizer_kwargs()
        optimizer = tf.train.AdagradOptimizer(**kwargs)
        optimizer = self.set_gradient_clipping(optimizer)

        return optimizer

    def validate_result(self):
        if self.gradient_clipping is None:
            assert isinstance(self.result, tf.train.AdagradOptimizer)

    @staticmethod
    def optimizer_type() -> str:
        return OptimizerStrategy.ADAGRAD_OPTIMIZER


class AdaDeltaOptimizerStrategy(OptimizerStrategy):
    """ AdaDelta optimizer, adaptive learning rates without decay.
    """
    def build_optimizer(self):
        kwargs = self.set_optimizer_kwargs()
        optimizer = tf.train.AdadeltaOptimizer(**kwargs)
        optimizer = self.set_gradient_clipping(optimizer)

        return optimizer

    def validate_result(self):
        if self.gradient_clipping is None:
            assert isinstance(self.result, tf.train.AdadeltaOptimizer)

    @staticmethod
    def optimizer_type() -> str:
        return OptimizerStrategy.ADADELTA_OPTIMIZER


class AdamOptimizerStrategy(OptimizerStrategy):
    """ Adam optimizer, adaptive momentum and adaptive learning rates.
    """
    def build_optimizer(self):
        kwargs = self.set_optimizer_kwargs()
        optimizer = tf.train.AdamOptimizer(**kwargs)
        optimizer = self.set_gradient_clipping(optimizer)

        return optimizer

    def validate_result(self):
        if self.gradient_clipping is None:
            assert isinstance(self.result, tf.train.AdamOptimizer)

    @staticmethod
    def optimizer_type() -> str:
        return OptimizerStrategy.ADAM_OPTIMIZER


class OptimizerStrategyFactory:
    strategies = [
        GradientDescentOptimizerStrategy,
        AdamOptimizerStrategy,
        AdaDeltaOptimizerStrategy,
        AdagradOptimizerStrategy
    ]  # type: List[OptimizerStrategy]

    @staticmethod
    def get_strategy(ml_model: AI,
                     optimizer_type: str,
                     learning_rate: float,
                     gradient_clipping: Optional[float] = None,
                     kwargs: dict = None):

        strategy_kwargs = {'ml_model': ml_model, 'learning_rate': learning_rate}
        if gradient_clipping is not None:
            strategy_kwargs['gradient_clipping'] = gradient_clipping

        if kwargs is not None:
            strategy_kwargs.update(kwargs)

        for strategy in OptimizerStrategyFactory.strategies:
            if optimizer_type == strategy.optimizer_type():
                return strategy(**strategy_kwargs)

        raise RuntimeError('Estimator type ({}) not found.'.format(optimizer_type))
