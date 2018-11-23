from abc import ABC, abstractmethod
from AIBuilder.Data import DataModel
import tensorflow as tf
from typing import Callable


# abstract AI class
class AbstractAI(ABC):

    def __init__(self):
        self.description = 'description not set.'

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
    def set_name(self, name: str):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def set_log_dir(self, path: str):
        pass

    @abstractmethod
    def get_log_dir(self) -> str:
        pass

    @abstractmethod
    def set_project_name(self, project_name: str):
        pass

    @abstractmethod
    def get_project_name(self) -> str:
        pass

    @abstractmethod
    def set_training_data(self, training_data: DataModel):
        pass

    @abstractmethod
    def get_training_data(self):
        pass

    @abstractmethod
    def set_training_fn(self, input_fn: Callable):
        pass

    @abstractmethod
    def set_evaluation_data(self, validation_data: DataModel):
        pass

    @abstractmethod
    def get_evaluation_data(self):
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

    def __init__(self, name: str, project_name: str, log_dir: str):
        super().__init__()
        self.project_name = project_name
        self.log_dir = log_dir
        self.name = name
        self.training_data = None
        self.evaluation_data = None

    def train(self):
        assert callable(self.training_fn), 'Training input Callable not set on AI: {}'\
            .format(self.__class__.__name__)

        self.estimator.train(input_fn=self.training_fn)

    def evaluate(self) -> dict:
        assert callable(self.evaluation_fn), 'Evaluation input Callable not set on AI: {}'\
            .format(self.__class__.__name__)

        return self.estimator.evaluate(input_fn=self.evaluation_fn)

    def estimate(self):
        pass

    def set_name(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    def set_log_dir(self, path: str):
        self.log_dir = path

    def get_log_dir(self) -> str:
        return self.log_dir

    def set_project_name(self, project_name: str):
        self.project_name = project_name

    def get_project_name(self) -> str:
        return self.project_name

    def set_training_data(self, training_data: DataModel):
        self.training_data = training_data

    def set_training_fn(self, input_fn: Callable):
        self.training_fn = input_fn

    def get_training_data(self):
        return self.evaluation_data

    def set_evaluation_data(self, evaluation_data: DataModel):
        self.evaluation_data = evaluation_data

    def get_evaluation_data(self):
        return self.evaluation_data

    def set_evaluation_fn(self, input_fn: Callable):
        self.evaluation_fn = input_fn

    def set_optimizer(self, optimizer: tf.train.Optimizer):
        self.optimizer = optimizer

    def set_estimator(self, estimator: tf.estimator):
        self.estimator = estimator

    def set_model_log_dir(self, model_dir: str):
        self.estimator.model_dir = model_dir
