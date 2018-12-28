from abc import ABC, abstractmethod
from typing import List
import tensorflow as tf
from AIBuilder.Data import DataModel


class FeatureColumnStrategy(ABC):
    INDICATOR_COLUMN_VOC_LIST = 'indicator_column'
    NUMERICAL_COLUMN = 'numeric_column'
    CATEGORICAL_COLUMN_VOC_LIST = 'categorical_column_with_vocabulary_list'
    MULTIPLE_HOT_COLUMNS = 'multiple_hot_columns'

    def __init__(self, column_name: str, data_model: DataModel):
        self.data_model = data_model
        self.column_name = column_name
        self.result = None

    def build(self) -> tf.feature_column:
        self.result = self.build_column()
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
        return tf.feature_column.numeric_column(self.column_name)

    def validate_result(self):
        assert self.result.__class__.__name__ == '_NumericColumn'

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.NUMERICAL_COLUMN]


class CategoricalColumnWithVOCListStrategy(FeatureColumnStrategy):

    def build_column(self) -> tf.feature_column:
        category_grabber = SimpleCategoryGrabber(data_model=self.data_model, column_name=self.column_name)
        categories = category_grabber.grab()

        return tf.feature_column.categorical_column_with_vocabulary_list(
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

        return tf.feature_column.indicator_column(categorical_column)

    def validate_result(self):
        assert self.result.__class__.__name__ == '_IndicatorColumn'

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.INDICATOR_COLUMN_VOC_LIST]


class MultipleHotFeatureStrategy(FeatureColumnStrategy):

    def build_column(self) -> tf.feature_column:
        df = self.data_model.get_dataframe()
        binary_feature_cols = []

        for column in df:
            if self.column_name in column:
                new_binary_feature_col = tf.feature_column.categorical_column_with_identity(
                    key=column,
                    num_buckets=2)
                binary_feature_cols.append(new_binary_feature_col)

        return binary_feature_cols

    def validate_result(self):
        assert type(self.result) is list
        for result in self.result:
            assert result.__class__.__name__ == '_IdentityCategoricalColumn'

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.MULTIPLE_HOT_COLUMNS]


class FeatureColumnStrategyFactory:
    strategies = [
        NumericColumnStrategy,
        CategoricalColumnWithVOCListStrategy,
        IndicatorColumnWithVOCListStrategy,
        MultipleHotFeatureStrategy
    ]  # type: List[FeatureColumnStrategy]

    @staticmethod
    def get_strategy(column_name: str, column_type: str, data_model: DataModel):

        for strategy in FeatureColumnStrategyFactory.strategies:
            if column_type in strategy.column_types():
                return strategy(column_name, data_model)

        raise RuntimeError('feature column type ({}) not found for column {}'.format(column_type, column_name))


class CategoryGrabber(ABC):

    def __init__(self, data_model: DataModel, column_name: str, exclusion_list: List[str] = None):
        self.data_model = data_model
        self.column_name = column_name
        self.validate_column()
        self.column = self.get_series()
        if exclusion_list is None:
            exclusion_list = []

        self.exclusion_list = exclusion_list

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
        categories: List = self.column.unique().tolist()
        for excluded_category in self.exclusion_list:
            categories.remove(excluded_category)

        return categories


class FromListCategoryGrabber(CategoryGrabber):

    def grab(self) -> list:
        categories = []
        for assigned_categories in self.column:
            for category in assigned_categories:
                if category not in categories and category not in self.exclusion_list:
                    categories.append(category)

        return categories
