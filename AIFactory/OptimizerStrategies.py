from abc import ABC, abstractmethod
from typing import List, Optional
import tensorflow as tf
from AIBuilder import AI


class OptimizerStrategy(ABC):
    GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent_optimizer'

    def __init__(self, ml_model: AI, learning_rate: int, gradient_clipping: Optional[int] = None, **kwargs):
        self.gradient_clipping = None
        if gradient_clipping is not None:
            self.gradient_clipping = gradient_clipping

        self.learning_rate = learning_rate
        self.ML_Model = ml_model
        self.kwargs = kwargs
        self.result = None

    def build(self) -> tf.estimator:
        self.result = self.build_optimizer()
        self.validate_result()

        return self.result

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

    def build_optimizer(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        optimizer = self.set_gradient_clipping(optimizer)

        return optimizer

    def validate_result(self):
        assert isinstance(self.result, tf.train.GradientDescentOptimizer)
        assert isinstance(self.result, GradientDescentOptimizerStrategy)

    @staticmethod
    def optimizer_type() -> str:
        return OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER


class OptimizerStrategyFactory:
    strategies = [
        GradientDescentOptimizerStrategy,
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
