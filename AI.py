from abc import ABC, abstractmethod
from AIBuilder.Data import DataModel
import tensorflow as tf


# abstract AI class
class AbstractAI(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def set_training_data(self, training_data: DataModel):
        pass

    @abstractmethod
    def set_validation_data(self, validation_data: DataModel):
        pass

    @abstractmethod
    def set_optimizer(self, optimizer: tf.train.Optimizer):
        pass

    @abstractmethod
    def set_estimator(self, estimator: tf.estimator):
        pass


class AI(AbstractAI):

    training_data: DataModel
    validation_data: DataModel
    estimator: tf.estimator
    optimizer: tf.train.Optimizer

    def train(self):
        pass

    def evaluate(self):
        pass

    def estimate(self):
        pass

    def set_training_data(self, training_data: DataModel):
        self.training_data = training_data

    def set_validation_data(self, validation_data: DataModel):
        self.validation_data = validation_data

    def set_optimizer(self, optimizer: tf.train.Optimizer):
        self.optimizer = optimizer

    def set_estimator(self, estimator: tf.estimator):
        self.estimator = estimator
