import unittest
from abc import ABC, abstractmethod
import pandas as pd
from AIBuilder.AIFactory.FeatureColumnStrategies import NumericColumnStrategy, FeatureColumnStrategy, \
    CategoricalColumnWithVOCListStrategy, IndicatorColumnWithVOCListStrategy, FeatureColumnStrategyFactory
from AIBuilder.Data import DataModel


class AbstractFCStrategyTester(ABC):
    TEST_COLUMN_NAME = 'col'

    def setUp(self):
        data = self.get_test_data()
        self.data_frame = pd.DataFrame(data=data)
        self.data_model = DataModel(self.data_frame)
        strategy_class = self.get_strategy_class_name()
        self.strategy = strategy_class(self.TEST_COLUMN_NAME, self.data_model)

    @property
    @abstractmethod
    def get_strategy_class_name(self) -> FeatureColumnStrategy:
        pass

    @property
    @abstractmethod
    def get_test_data(self) -> dict:
        pass

    def testBuild(self):
        result = self.strategy.build()
        self.assert_result(result)

    @abstractmethod
    def assert_result(self, result):
        pass


class TestNumericColumnStrategy(AbstractFCStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> FeatureColumnStrategy:
        return NumericColumnStrategy

    def get_test_data(self):
        return {}

    def assert_result(self, result):
        self.assertIsNotNone(result)


class TestCategoricalColumnWithVOCListStrategy(AbstractFCStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> FeatureColumnStrategy:
        return CategoricalColumnWithVOCListStrategy

    def get_test_data(self):
        return {'col': ['cat1', 'cat2', 'cat1']}

    def assert_result(self, result):
        self.assertIsNotNone(result)
        self.assertListEqual(list(result.vocabulary_list), ['cat1', 'cat2'])


class TestIndicatorColumnWithVOCListStrategy(AbstractFCStrategyTester, unittest.TestCase):

    def get_strategy_class_name(self) -> FeatureColumnStrategy:
        return IndicatorColumnWithVOCListStrategy

    def get_test_data(self):
        return {'col': [['cat1', 'cat2'], ['cat1', 'cat3']]}

    def assert_result(self, result):
        self.assertIsNotNone(result)
        self.assertListEqual(list(result.categorical_column.vocabulary_list), ['cat1', 'cat2', 'cat3'])


class TestFeatureColumnStrategyFactory(unittest.TestCase):

    def setUp(self):
        data = []
        self.data_frame = pd.DataFrame(data=data)
        self.data_model = DataModel(self.data_frame)

    def test_numerical_column(self):
        strategy = FeatureColumnStrategyFactory.get_strategy('col1',
                                                             FeatureColumnStrategy.NUMERICAL_COLUMN,
                                                            self.data_model)

        self.assertIsInstance(strategy, NumericColumnStrategy)

    def test_categorical_column(self):
        strategy = FeatureColumnStrategyFactory.get_strategy('col1',
                                                             FeatureColumnStrategy.CATEGORICAL_COLUMN_VOC_LIST,
                                                            self.data_model)

        self.assertIsInstance(strategy, CategoricalColumnWithVOCListStrategy)

    def test_indicator_column(self):
        strategy = FeatureColumnStrategyFactory.get_strategy('col1',
                                                             FeatureColumnStrategy.INDICATOR_COLUMN_VOC_LIST,
                                                            self.data_model)

        self.assertIsInstance(strategy, IndicatorColumnWithVOCListStrategy)
