import unittest
from unittest import mock
from abc import ABC, abstractmethod
from AIBuilder.AI import AI
from AIBuilder.AIFactory.EstimatorStrategies import EstimatorStrategy, LinearRegressorStrategy, EstimatorStrategyFactory


class AbstractEstimatorStrategyTester(ABC):

    def setUp(self):
        strategy_class = self.get_strategy_class_name()

        self.ml_model = AI(project_name='test', log_dir='test/dir', name='test')

        self.ml_model.optimizer = mock.Mock()
        self.ml_model.training_data = mock.Mock()
        self.ml_model.training_data.get_tf_feature_columns = mock.Mock()
        self.strategy = strategy_class(self.ml_model)

    @property
    @abstractmethod
    def get_strategy_class_name(self) -> EstimatorStrategy:
        pass

    def testBuild(self):
        result = self.strategy.build_estimator()
        self.assert_result(result)

    @abstractmethod
    def assert_result(self, result):
        pass


class TestLinearRegressorStrategy(AbstractEstimatorStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> EstimatorStrategy:
        return LinearRegressorStrategy

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestEstimatorStrategyFactory(unittest.TestCase):

    def setUp(self):
        self.ml_model = AI(project_name='test', log_dir='test/dir', name='test')

    def test_LinearRegressor(self):
        strategy = EstimatorStrategyFactory.get_strategy(self.ml_model, EstimatorStrategy.LINEAR_REGRESSOR)
        self.assertIsInstance(strategy, LinearRegressorStrategy)
