import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Union
import tensorflow as tf
from AIBuilder.Data import DataModel


class FeatureColumnStrategy(ABC):
    """ Strategy to build one or more tf.feature column for the ml model."""

    CATEGORICAL_COLUMN_IDENTITY = 'categorical_column_with_identity'
    INDICATOR_COLUMN_VOC_LIST = 'indicator_column'
    NUMERICAL_COLUMN = 'numeric_column'
    CATEGORICAL_COLUMN_VOC_LIST = 'categorical_column_with_vocabulary_list'
    MULTIPLE_HOT_COLUMNS = 'multiple_hot_columns'
    VECTOR_COLUMNS = 'vector_columns'
    BUCKETIZED_COLUMN = 'bucketized_column'
    CROSSED_COLUMN = 'crossed_column'

    ALL_COLUMNS = [CATEGORICAL_COLUMN_IDENTITY, CATEGORICAL_COLUMN_VOC_LIST, NUMERICAL_COLUMN,
                   INDICATOR_COLUMN_VOC_LIST,
                   MULTIPLE_HOT_COLUMNS, BUCKETIZED_COLUMN, CROSSED_COLUMN, VECTOR_COLUMNS]

    """ Results of building process will be stored in this attribute. """
    results: Union[None, list]

    def __init__(self, column_name: str, data_model: DataModel, feature_config: dict = None):
        """

        Args:
            column_name: Name of column to build feature model for.
            data_model: Data model to build feature column for and finally add feature column to.
            feature_config: Various Data that can be passed to the strategy.
        """
        self.feature_config = feature_config
        self.data_model = data_model
        self.column_name = column_name
        self.results = None

    def build(self) -> list:
        """ Run building process, called by factory."""
        self.results = self.build_column()
        self.validate_result()

        return self.results

    @abstractmethod
    def build_column(self) -> list:
        """
        Build the column(s) in this method and return it/them in a list.

        Returns: list

        """
        pass

    @abstractmethod
    def validate_result(self):
        """
        Validate the result of the building process.

        This method is ran after building the feature columns, the 'result' attribute should contain the output of
        build_column.

        Returns:

        """
        pass

    @staticmethod
    @abstractmethod
    def column_types() -> list:
        """ Return list of feature column types, usually just one.
        Returns: list

        """
        pass


class NumericColumnStrategy(FeatureColumnStrategy):

    def build_column(self) -> list:
        return [tf.feature_column.numeric_column(self.column_name)]

    def validate_result(self):
        for result in self.results:
            assert result.__class__.__name__ in ['NumericColumn',
                                                 '_NumericColumn'], 'Result is wrong class "{}"'.format(
                result.__class__.__name__)

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.NUMERICAL_COLUMN]


class CategoricalColumnWithVOCListStrategy(FeatureColumnStrategy):

    def build_column(self) -> list:
        category_grabber = SimpleCategoryGrabber(data_model=self.data_model, column_name=self.column_name)
        categories = category_grabber.grab()

        new_column = tf.feature_column.categorical_column_with_vocabulary_list(
            self.column_name,
            vocabulary_list=categories
        )

        return [new_column]

    def validate_result(self):
        for result in self.results:
            assert result.__class__.__name__ in ['_VocabularyListCategoricalColumn',
                                                 'VocabularyListCategoricalColumn'], 'Result is wrong class "{}"' \
                .format(result.__class__.__name__)

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.CATEGORICAL_COLUMN_VOC_LIST]


class CategoricalColumnWithIdentity(FeatureColumnStrategy):

    def build_column(self) -> list:
        df = self.data_model.get_dataframe()
        num_buckets = len(df[self.column_name].unique())

        new_column = tf.feature_column.categorical_column_with_identity(
            key=self.column_name,
            num_buckets=num_buckets)

        return [new_column]

    def validate_result(self):
        for result in self.results:
            assert result.__class__.__name__ == 'IdentityCategoricalColumn', 'Result is wrong class "{}"'.format(
                result.__class__.__name__)

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.CATEGORICAL_COLUMN_IDENTITY]


class IndicatorColumnWithVOCListStrategy(FeatureColumnStrategy):

    def build_column(self) -> list:
        # todo: which grabber to use here?
        category_grabber = SimpleCategoryGrabber(data_model=self.data_model, column_name=self.column_name)
        categories = category_grabber.grab()

        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
            self.column_name,
            vocabulary_list=categories
        )

        indicator_column = tf.feature_column.indicator_column(categorical_column)

        return [indicator_column]

    def validate_result(self):
        for result in self.results:
            assert result.__class__.__name__ in ['IndicatorColumn', '_IndicatorColumn'], \
                'Result is wrong class "{}"'.format(result.__class__.__name__)

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.INDICATOR_COLUMN_VOC_LIST]


class MultipleHotFeatureStrategy(FeatureColumnStrategy):
    """ This strategy adds a categorical feature column for every category found in the data column.

    The categorical features have two categories 0 and 1 and are named after their respective catagory prefixed with
    the name of the original column.

    example:
     input:
        pets: ['cat', 'dog', 'bird'], ['cat', 'dog'], ['cat']
        pets_cat: [1,1,1]
        pets_dog: [1,1,0]
        pets_bird: [1,0,0]

     adds:
        pets_cat: 'categorical_column_with_identity'
        pets_dog: 'categorical_column_with_identity'
        pets_bird: 'categorical_column_with_identity'

    note: The data in 'pets' can be scrubbed by the 'MultipleCatListToMultipleHotScrubber' to create the required
    data structure.
    """

    def build_column(self) -> list:
        df = self.data_model.get_dataframe()
        binary_feature_cols = []

        column_prefix = self.column_name + '_'
        for column in df:
            if column_prefix in column:
                new_binary_feature_col = tf.feature_column.categorical_column_with_identity(
                    key=column,
                    num_buckets=2)
                binary_feature_cols.append(new_binary_feature_col)

        return binary_feature_cols

    def validate_result(self):
        for result in self.results:
            assert result.__class__.__name__ == 'IdentityCategoricalColumn'

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.MULTIPLE_HOT_COLUMNS]


class VectorColumnsStrategy(FeatureColumnStrategy):
    """
        Adds multiple numeric feature columns for vectors that are being stored in multiple columns.
    """

    NAME_CONFIG_KEY = 'col_name'

    def build_column(self):
        df = self.data_model.get_dataframe()
        all_columns = df.columns

        feature_columns = [column for column in all_columns if self.feature_config[self.NAME_CONFIG_KEY] in column]
        columns = [tf.feature_column.numeric_column(column) for column in feature_columns]

        return columns

    def validate_result(self):
        pass

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.VECTOR_COLUMNS]


def bucketize_data(column_data: pd.Series, num_buckets: int) -> List[int]:
    min_val = column_data.min()
    max_val = column_data.max()
    bucket_size = (max_val - min_val) / num_buckets
    boundries = []
    boundry = min_val
    while (len(boundries) + 1) < num_buckets:
        boundry += bucket_size
        boundries.append(round(boundry + 0.000001))

    return boundries


class BucketizedColumnStrategy(FeatureColumnStrategy):

    def build_column(self) -> list:
        num_feature_column = tf.feature_column.numeric_column(self.column_name)
        column_data = self.data_model.get_dataframe()[self.column_name]

        boundries = buckets = self.feature_config['buckets']
        if type(buckets) is int:
            boundries = bucketize_data(column_data, buckets)

        bucketized_column = tf.feature_column.bucketized_column(
            source_column=num_feature_column,
            boundaries=boundries
        )

        return [bucketized_column]

    def validate_result(self):
        for result in self.results:
            assert result.__class__.__name__ in ['BucketizedColumn', '_BucketizedColumn'], \
                'Result is wrong class "{}"'.format(result.__class__.__name__)

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.BUCKETIZED_COLUMN]


class CrossedColumnStrategy(FeatureColumnStrategy):

    def build_column(self):
        current_columns = self.data_model.get_tf_feature_columns()

        assert 'columns' in self.feature_config, 'columns config key not passed to crossed column strategy.'
        assert 'num_buckets' in self.feature_config, 'num_bucket config key not passed to crossed column strategy.'

        column_names = self.feature_config['columns']
        num_buckets = self.feature_config['num_buckets']

        columns_to_cross = []
        for name in column_names:
            for column in current_columns:
                if name in column.name:
                    columns_to_cross.append(column)

        assert 2 is len(columns_to_cross), 'Two columns must be crossed found {}, looked for "{}" in "{}"' \
            .format(len(columns_to_cross), column_names, current_columns)

        crossed_tf_column = tf.feature_column.crossed_column(columns_to_cross, num_buckets)

        return [crossed_tf_column]

    def validate_result(self):
        for result in self.results:
            assert result.__class__.__name__ == 'CrossedColumn'

    @staticmethod
    def column_types() -> list:
        return [FeatureColumnStrategy.CROSSED_COLUMN]


class FeatureColumnStrategyFactory:
    strategies = [
        NumericColumnStrategy,
        CategoricalColumnWithIdentity,
        CategoricalColumnWithVOCListStrategy,
        IndicatorColumnWithVOCListStrategy,
        MultipleHotFeatureStrategy,
        VectorColumnsStrategy,
        BucketizedColumnStrategy,
        CrossedColumnStrategy,
    ]  # type: List[FeatureColumnStrategy]

    @staticmethod
    def get_strategy(column_name: str, column_type: str, data_model: DataModel, feature_config: dict = None):
        if feature_config is None:
            feature_config = {}

        for strategy in FeatureColumnStrategyFactory.strategies:
            if column_type in strategy.column_types():
                column_feature_config = {}
                if column_name in feature_config:
                    column_feature_config = feature_config[column_name]

                return strategy(column_name, data_model, column_feature_config)

        raise RuntimeError(f'feature column type ("{column_type}") not found for column "{column_name}".')


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
