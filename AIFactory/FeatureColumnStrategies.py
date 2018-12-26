from abc import ABC, abstractmethod
from typing import List
import tensorflow as tf
from AIBuilder.Data import DataModel


class FeatureColumnStrategy(ABC):
    INDICATOR_COLUMN_VOC_LIST = 'indicator_column'
    NUMERICAL_COLUMN = 'numeric_column'
    CATEGORICAL_COLUMN_VOC_LIST = 'categorical_column_with_vocabulary_list'

    def __init__(self, column_name: str, data_model: DataModel):
        self.data_model = data_model
        self.column_name = column_name
        self.result = None

    def build(self) -> tf.feature_column:
        self.build_column()
        self.validate_result()

        return self.result

    @abstractmethod
    def build_column(self) -> tf.feature_column:
        pass

    @abstractmethod
    def validate_result(self):
        pass

    @staticmethod
    @abstractmethod
    def column_types() -> list:
        pass


class NumericColumnStrategy(FeatureColumnStrategy):

    def build_column(self) -> tf.feature_column:
        self.result = tf.feature_column.numeric_column(self.column_name)

    def validate_result(self):
        assert self.result.__class__.__name__ == '_NumericColumn'

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.NUMERICAL_COLUMN]


class CategoricalColumnWithVOCListStrategy(FeatureColumnStrategy):

    def build_column(self) -> tf.feature_column:
        category_grabber = SimpleCategoryGrabber(data_model=self.data_model, column_name=self.column_name)
        categories = category_grabber.grab()

        self.result = tf.feature_column.categorical_column_with_vocabulary_list(
            self.column_name,
            vocabulary_list=categories
        )

    def validate_result(self):
        assert self.result.__class__.__name__ == '_VocabularyListCategoricalColumn'

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.CATEGORICAL_COLUMN_VOC_LIST]


class IndicatorColumnWithVOCListStrategy(FeatureColumnStrategy):

    def build_column(self) -> tf.feature_column:
        category_grabber = FromListCategoryGrabber(data_model=self.data_model, column_name=self.column_name)
        categories = category_grabber.grab()

        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
            self.column_name,
            vocabulary_list=categories
        )

        self.result = tf.feature_column.indicator_column(categorical_column)

    def validate_result(self):
        assert self.result.__class__.__name__ == '_IndicatorColumn'

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.INDICATOR_COLUMN_VOC_LIST]


class FeatureColumnStrategyFactory:
    strategies = [
        NumericColumnStrategy,
        CategoricalColumnWithVOCListStrategy,
        IndicatorColumnWithVOCListStrategy,
    ]  # type: List[FeatureColumnStrategy]

    @staticmethod
    def get_strategy(column_name: str, column_type: str, data_model: DataModel):

        for strategy in FeatureColumnStrategyFactory.strategies:
            if column_type in strategy.column_types():
                return strategy(column_name, data_model)

        raise RuntimeError('feature column type ({}) not found for column {}'.format(column_type, column_name))


class CategoryGrabber(ABC):
    def __init__(self, data_model: DataModel, column_name: str):
        self.data_model = data_model
        self.column_name = column_name
        self.validate_column()
        self.column = self.get_series()

    def validate_column(self):
        df = self.data_model.get_dataframe()
        assert self.column_name in df, 'Column {} not in dataframe.'.format(self.column_name)

    def get_series(self):
        df = self.data_model.get_dataframe()
        return df[self.column_name]

    @abstractmethod
    def grab(self) -> list:
        pass


class SimpleCategoryGrabber(CategoryGrabber):

    def grab(self) -> list:
        return self.column.unique().tolist()


class FromListCategoryGrabber(CategoryGrabber):

    def grab(self) -> list:
        categories = []
        for assigned_categories in self.column:
            for category in assigned_categories:
                if category not in categories:
                    categories.append(category)

        return categories
