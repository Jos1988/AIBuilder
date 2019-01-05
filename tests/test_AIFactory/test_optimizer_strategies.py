import unittest
from unittest import mock
from abc import ABC, abstractmethod
from AIBuilder.AI import AI
from AIBuilder.AIFactory.OptimizerStrategies import GradientDescentOptimizerStrategy, OptimizerStrategy, \
    OptimizerStrategyFactory, AdagradOptimizerStrategy, AdaDeltaOptimizerStrategy, AdamOptimizerStrategy


class AbstractOptimizerStrategyTester(ABC):

    def setUp(self):
        strategy_class = self.get_strategy_class_name()

        self.ml_model = AI(project_name='test', log_dir='test/dir', name='test')

        self.ml_model.optimizer = mock.Mock()
        self.ml_model.training_data = mock.Mock()
        self.ml_model.training_data.get_tf_feature_columns = mock.Mock()

        kwargs = {'ml_model': self.ml_model, 'learning_rate': 0.1}
        kwargs.update(self.additional_parameters())

        self.strategy = strategy_class(**kwargs)

    @property
    @abstractmethod
    def get_strategy_class_name(self) -> OptimizerStrategy:
        pass

    def testBuild(self):
        result = self.strategy.build()
        self.assert_result(result)

    def additional_parameters(self):
        return {}

    @abstractmethod
    def assert_result(self, result):
        pass


class TestGradientDescentOptimizerStrategy1(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return GradientDescentOptimizerStrategy

    def additional_parameters(self):
        return {'gradient_clipping': 5}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestGradientDescentOptimizerStrategy2(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return GradientDescentOptimizerStrategy

    def additional_parameters(self):
        return {}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestGradientDescentOptimizerStrategy3(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return GradientDescentOptimizerStrategy

    def additional_parameters(self):
        return {'gradient_clipping': 5, 'some_arg': 1}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestGradientDescentOptimizerStrategy4(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return GradientDescentOptimizerStrategy

    def additional_parameters(self):
        return {'some_arg': 1}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestAdagradOptimizerStrategy1(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return AdagradOptimizerStrategy

    def additional_parameters(self):
        return {'gradient_clipping': 5}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestAdagradOptimizerStrategy2(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return AdagradOptimizerStrategy

    def additional_parameters(self):
        return {}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestAdaDeltaOptimizerStrategy1(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return AdaDeltaOptimizerStrategy

    def additional_parameters(self):
        return {'gradient_clipping': 5}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestAdaDeltaOptimizerStrategy2(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return AdaDeltaOptimizerStrategy

    def additional_parameters(self):
        return {}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestAdamOptimizerStrategy1(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return AdamOptimizerStrategy

    def additional_parameters(self):
        return {'gradient_clipping': 5}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestAdamOptimizerStrategy2(AbstractOptimizerStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> OptimizerStrategy:
        return AdamOptimizerStrategy

    def additional_parameters(self):
        return {}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestOptimizerStrategyFactory(unittest.TestCase):

    def setUp(self):
        self.ml_model = AI(project_name='test', log_dir='test/dir', name='test')

    def test_GradientDescentOptimizer1(self):
        strategy = OptimizerStrategyFactory.get_strategy(
            ml_model=self.ml_model,
            optimizer_type=OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=0.1,
            gradient_clipping=5,
            kwargs={'arg': 1})

        self.assertIsInstance(strategy, GradientDescentOptimizerStrategy)

    def test_GradientDescentOptimizer2(self):
        strategy = OptimizerStrategyFactory.get_strategy(
            ml_model=self.ml_model,
            optimizer_type=OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=0.1,
            gradient_clipping=5)

        self.assertIsInstance(strategy, GradientDescentOptimizerStrategy)

    def test_GradientDescentOptimizer3(self):
        strategy = OptimizerStrategyFactory.get_strategy(
            ml_model=self.ml_model,
            optimizer_type=OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=0.1,
            kwargs={'arg': 1})

        self.assertIsInstance(strategy, GradientDescentOptimizerStrategy)

    def test_GradientDescentOptimizer4(self):
        strategy = OptimizerStrategyFactory.get_strategy(
            ml_model=self.ml_model,
            optimizer_type=OptimizerStrategy.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=0.1)

        self.assertIsInstance(strategy, GradientDescentOptimizerStrategy)

    def test_AdagradOptimizer4(self):
        strategy = OptimizerStrategyFactory.get_strategy(
            ml_model=self.ml_model,
            optimizer_type=OptimizerStrategy.ADAGRAD_OPTIMIZER,
            learning_rate=0.1)

        self.assertIsInstance(strategy, AdagradOptimizerStrategy)

    def test_AdadeltaDescentOptimizer4(self):
        strategy = OptimizerStrategyFactory.get_strategy(
            ml_model=self.ml_model,
            optimizer_type=OptimizerStrategy.ADADELTA_OPTIMIZER,
            learning_rate=0.1)

        self.assertIsInstance(strategy, AdaDeltaOptimizerStrategy)

    def test_AdamOptimizer4(self):
        strategy = OptimizerStrategyFactory.get_strategy(
            ml_model=self.ml_model,
            optimizer_type=OptimizerStrategy.ADAM_OPTIMIZER,
            learning_rate=0.1)

        self.assertIsInstance(strategy, AdamOptimizerStrategy)
