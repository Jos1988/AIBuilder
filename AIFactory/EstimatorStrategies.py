from abc import ABC, abstractmethod
from typing import List
import tensorflow as tf
from AIBuilder import AI


class EstimatorStrategy(ABC):
    LINEAR_REGRESSOR = 'linear_regressor'
    DNN_REGRESSOR = 'dnn_regressor'

    def __init__(self, ml_model: AI):
        self.ML_Model = ml_model
        self.result = None

    def build(self) -> tf.estimator:
        self.result = self.build_estimator()
        self.validate_result()

        return self.result

    def get_model_dir(self) -> str:
        return self.ML_Model.get_log_dir() + '/' + self.ML_Model.get_project_name() + '/tensor_board/' + \
               self.ML_Model.get_name()

    @abstractmethod
    def build_estimator(self) -> tf.feature_column:
        pass

    @abstractmethod
    def validate_result(self):
        pass

    @staticmethod
    @abstractmethod
    def estimator_type() -> str:
        pass


class LinearRegressorStrategy(EstimatorStrategy):

    def build_estimator(self) -> tf.feature_column:
        estimator = tf.estimator.LinearRegressor(
            feature_columns=self.ML_Model.training_data.get_tf_feature_columns(),
            optimizer=self.ML_Model.optimizer,
            model_dir=self.get_model_dir()
        )

        return estimator

    def validate_result(self):
        assert isinstance(self.result, tf.estimator.LinearRegressor)

    @staticmethod
    def estimator_type() -> str:
        return EstimatorStrategy.LINEAR_REGRESSOR


class DNNRegressorStrategy(EstimatorStrategy):

    def build_estimator(self) -> tf.feature_column:
        estimator = tf.estimator.DNNRegressor(
            feature_columns=self.ML_Model.training_data.get_tf_feature_columns(),
            hidden_units=[1024, 512, 256],
            optimizer=self.ML_Model.optimizer,
            model_dir=self.get_model_dir(),
        )

        return estimator

    def validate_result(self):
        assert isinstance(self.result, tf.estimator.DNNRegressor)

    @staticmethod
    def estimator_type() -> str:
        return EstimatorStrategy.DNN_REGRESSOR


# Is there a bridge pattern here?
class EstimatorStrategyFactory:
    strategies = [
        LinearRegressorStrategy,
    ]  # type: List[EstimatorStrategy]

    @staticmethod
    def get_strategy(ml_model: AI, estimator_type: str):

        for strategy in EstimatorStrategyFactory.strategies:
            if estimator_type == strategy.estimator_type():
                return strategy(ml_model)

        raise RuntimeError('Estimator type ({}) not found.'.format(estimator_type))
