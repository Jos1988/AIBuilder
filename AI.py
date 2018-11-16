from abc import ABC, abstractmethod
from AIBuilder.Data import DataModel
import tensorflow as tf
from typing import Callable


# abstract AI class
class AbstractAI(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self, model_dir: str) -> dict:
        pass

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def set_training_data(self, training_data: DataModel):
        pass

    @abstractmethod
    def set_training_fn(self, input_fn: Callable):
        pass

    @abstractmethod
    def set_evaluation_data(self, validation_data: DataModel):
        pass

    @abstractmethod
    def set_evaluation_fn(self, input_fn: Callable):
        pass

    @abstractmethod
    def set_optimizer(self, optimizer: tf.train.Optimizer):
        pass

    @abstractmethod
    def set_estimator(self, estimator: tf.estimator):
        pass


class AI(AbstractAI):

    training_data: DataModel
    training_fn: Callable
    evaluation_data: DataModel
    evaluation_fn: Callable
    estimator: tf.estimator
    optimizer: tf.train.Optimizer

    def train(self):
        if type(self.training_fn) is not Callable:
            raise RuntimeError('Training input Callable not set on AI: {}'.format(self.__class__.__name__))

        self.estimator.train(input_fn=self.training_fn)

    def evaluate(self, model_dir: str) -> dict:
        if type(self.evaluation_fn) is not Callable:
            raise RuntimeError('Evaluation input Callable not set on AI: {}'.format(self.__class__.__name__))

        self.estimator.model_dir = model_dir
        return self.estimator.evaluate(input_fn=self.evaluation_fn)

    def estimate(self):
        pass

    def set_training_data(self, training_data: DataModel):
        self.training_data = training_data

    def set_training_fn(self, input_fn: Callable):
        self.training_fn = input_fn

    def set_evaluation_data(self, validation_data: DataModel):
        self.evaluation_data = validation_data

    def set_evaluation_fn(self, input_fn: Callable):
        self.evaluation_fn = input_fn

    def set_optimizer(self, optimizer: tf.train.Optimizer):
        self.optimizer = optimizer

    def set_estimator(self, estimator: tf.estimator):
        self.estimator = estimator

    def set_model_log_dir(self, model_dir: str):
        self.estimator.model_dir = model_dir
