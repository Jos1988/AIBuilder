from abc import ABC, abstractmethod
from typing import List
import tensorflow as tf
from AIBuilder import AI


class EstimatorStrategy(ABC):
    LINEAR_REGRESSOR = 'linear_regressor'
    DNN_REGRESSOR = 'dnn_regressor'

    def __init__(self, ml_model: AI, **kwargs):
        self.ML_Model = ml_model
        self.kwargs = kwargs
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
        kwargs = {'feature_columns': self.ML_Model.training_data.get_tf_feature_columns(),
                  'optimizer': self.ML_Model.optimizer,
                  'model_dir': self.get_model_dir(), }

        kwargs.update(self.kwargs)
        self.validate_kwargs(kwargs)
        estimator = tf.estimator.DNNRegressor(**kwargs)

        return estimator

    @staticmethod
    def kwarg_requirements() -> dict:
        return {'hidden_units': list}

    def validate_kwargs(self, kwargs):
        kwarg_min_requirements = self.kwarg_requirements()
        for key, data_type in kwarg_min_requirements.items():
            assert key in kwargs, 'DNNRegressor missing requires argument: {}.'.format(key)
            assert type(kwargs[key]) is data_type, 'DNNRegressor argument for {} must be {}, {} given ({})' \
                .format(key, data_type, type(kwargs[key]), kwargs[key])

    def validate_result(self):
        assert isinstance(self.result, tf.estimator.DNNRegressor)

    @staticmethod
    def estimator_type() -> str:
        return EstimatorStrategy.DNN_REGRESSOR


class EstimatorStrategyFactory:
    strategies = [
        LinearRegressorStrategy,
        DNNRegressorStrategy
    ]  # type: List[EstimatorStrategy]

    @staticmethod
    def get_strategy(ml_model: AI, estimator_type: str, kwargs: dict = None):

        strategy_kwargs = {'ml_model': ml_model}
        if kwargs is not None:
            strategy_kwargs.update(kwargs)

        for strategy in EstimatorStrategyFactory.strategies:
            if estimator_type == strategy.estimator_type():
                return strategy(**strategy_kwargs)

        raise RuntimeError('Estimator type ({}) not found.'.format(estimator_type))
