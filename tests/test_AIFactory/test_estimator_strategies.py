import unittest
from unittest import mock
from abc import ABC, abstractmethod
from AIBuilder.AI import AI
from AIBuilder.AIFactory.EstimatorStrategies import EstimatorStrategy, LinearRegressorStrategy, \
    EstimatorStrategyFactory, DNNRegressorStrategy


class AbstractEstimatorStrategyTester(ABC):

    def setUp(self):
        strategy_class = self.get_strategy_class_name()

        self.ml_model = AI(project_name='test', log_dir='test/dir', name='test')

        self.ml_model.optimizer = mock.Mock()
        self.ml_model.training_data = mock.Mock()
        self.ml_model.training_data.get_tf_feature_columns = mock.Mock()

        kwargs = {'ml_model': self.ml_model}
        kwargs.update(self.additional_parameters())

        self.strategy = strategy_class(**kwargs)

    @property
    @abstractmethod
    def get_strategy_class_name(self) -> EstimatorStrategy:
        pass

    def testBuild(self):
        result = self.strategy.build_estimator()
        self.assert_result(result)

    def additional_parameters(self):
        return {}

    @abstractmethod
    def assert_result(self, result):
        pass


class TestLinearRegressorStrategy(AbstractEstimatorStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> EstimatorStrategy:
        return LinearRegressorStrategy

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestDNNRegressorStrategy(AbstractEstimatorStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> EstimatorStrategy:
        return DNNRegressorStrategy

    def additional_parameters(self):
        return {'hidden_units': [5, 10, 5]}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestDNNRegressorStrategy2(AbstractEstimatorStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> EstimatorStrategy:
        return DNNRegressorStrategy

    def additional_parameters(self):
        return {'hidden_units': 'should be list'}

    def testBuild(self):
        with self.assertRaises(AssertionError):
            result = self.strategy.build_estimator()
            self.assert_result(result)

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestDNNRegressorStrategy3(AbstractEstimatorStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> EstimatorStrategy:
        return DNNRegressorStrategy

    def additional_parameters(self):
        return {}

    def testBuild(self):
        with self.assertRaises(AssertionError):
            result = self.strategy.build_estimator()
            self.assert_result(result)

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestEstimatorStrategyFactory(unittest.TestCase):

    def setUp(self):
        self.ml_model = AI(project_name='test', log_dir='test/dir', name='test')

    def test_LinearRegressor(self):
        strategy = EstimatorStrategyFactory.get_strategy(self.ml_model, EstimatorStrategy.LINEAR_REGRESSOR)
        self.assertIsInstance(strategy, LinearRegressorStrategy)

    def test_DNNRegressor(self):
        additional_arguments = {'test': 1}
        strategy = EstimatorStrategyFactory.get_strategy(self.ml_model,
                                                         EstimatorStrategy.DNN_REGRESSOR,
                                                         kwargs=additional_arguments)

        self.assertIsInstance(strategy, DNNRegressorStrategy)
        self.assertDictEqual(strategy.kwargs, additional_arguments)
