from abc import ABC, abstractmethod
from AIBuilder.Data import DataModel
import tensorflow as tf
from typing import Callable, Optional


# abstract AI class
class AbstractAI(ABC):

    def __init__(self):
        self.description = 'description not set.'
        self.results = {}

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        pass

    @abstractmethod
    def predict(self):
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
    def get_training_data(self) -> Optional[DataModel]:
        pass

    @abstractmethod
    def set_training_fn(self, input_fn: Callable):
        pass

    @abstractmethod
    def set_evaluation_data(self, validation_data: DataModel):
        pass

    @abstractmethod
    def get_evaluation_data(self) -> Optional[DataModel]:
        pass

    @abstractmethod
    def set_evaluation_fn(self, input_fn: Callable):
        pass

    @abstractmethod
    def set_prediction_data(self, prediction_data: DataModel):
        pass

    @abstractmethod
    def get_prediction_data(self):
        pass

    @abstractmethod
    def set_prediction_fn(self, input_fn: Callable):
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
    prediction_data: DataModel
    prediction_fn: Callable
    estimator: tf.estimator
    optimizer: tf.train.Optimizer
    results: dict

    def __init__(self, project_name: str, log_dir: str, name: str = None):
        super().__init__()
        self.project_name = project_name
        self.log_dir = log_dir
        self.name = name
        self.training_data = None
        self.evaluation_data = None
        self.prediction_data = None

    def train(self):
        assert callable(self.training_fn), 'Training input Callable not set on AI: {}'\
            .format(self.__class__.__name__)

        self.estimator.train(input_fn=self.training_fn)

    def evaluate(self):
        assert callable(self.evaluation_fn), 'Evaluation input Callable not set on AI: {}'\
            .format(self.__class__.__name__)

        self.results = self.estimator.evaluate(input_fn=self.evaluation_fn)

    def predict(self):
        return self.estimator.predict(input_fn=self.prediction_fn)

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
        return self.training_data

    def set_evaluation_data(self, evaluation_data: DataModel):
        self.evaluation_data = evaluation_data

    def get_evaluation_data(self):
        return self.evaluation_data

    def set_evaluation_fn(self, input_fn: Callable):
        self.evaluation_fn = input_fn

    def set_prediction_data(self, prediction_data: DataModel):
        self.prediction_data = prediction_data

    def get_prediction_data(self):
        return self.prediction_data

    def set_prediction_fn(self, input_fn: Callable):
        self.prediction_fn = input_fn

    def set_optimizer(self, optimizer: tf.train.Optimizer):
        self.optimizer = optimizer

    def set_estimator(self, estimator: tf.estimator):
        self.estimator = estimator
