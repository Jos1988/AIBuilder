import unittest
from AIBuilder.AI import AbstractAI, AI
from AIBuilder.AIFactory.Builders.Builder import Builder
from AIBuilder.Data import MetaData, DataModel, DataLoader, DataSetSplitter
from AIBuilder.AIFactory.Specifications.BasicSpecifications import RangeSpecification, DataTypeSpecification
from AIBuilder.AIFactory.Specifications.FeatureSpecifications import FeatureColumnsSpecification
import tensorflow as tf


class DataBuilder(Builder):
    CATEGORICAL_COLUMN_VOC_LIST = 'categorical_column_with_vocabulary_list'
    NUMERICAL_COLUMN = 'numeric_column'
    valid_column_types = [CATEGORICAL_COLUMN_VOC_LIST, NUMERICAL_COLUMN]

    def __init__(self, data_source: str,
                 target_column: str,
                 validation_data_percentage: int,
                 feature_columns: dict,
                 data_columns: list,
                 metadata: MetaData):

        self.data_columns = data_columns
        self.metadata = metadata
        self.validation_data_percentage = RangeSpecification('validation_data_perc', validation_data_percentage, 0, 100)
        self.training_data_percentage = RangeSpecification(name='training_data_perc',
                                                           value=(100 - self.validation_data_percentage.value),
                                                           min_value=0,
                                                           max_value=100)

        self.data_source = DataTypeSpecification('data_source', data_source, str)
        self.target_column = DataTypeSpecification('target_column', target_column, str)
        self.feature_columns = FeatureColumnsSpecification('feature_columns', [], self.valid_column_types)
        self.test_data = None
        self.validation_data = None

        # validate input.
        for name, type in feature_columns.items():
            self.add_feature_column(name=name, column_type=type)

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def builder_type(self) -> str:
        return self.DATA_MODEL

    def add_feature_column(self, name: str, column_type: str):
        self.feature_columns.add_feature_column(name=name, column_type=column_type)

    def validate(self):
        self.validate_specifications()
        assert self.target_column not in self.get_feature_column_names(), \
            'target column {}, also set as feature column!'.format(self.target_column())

    def build(self, ai: AbstractAI):
        data = self.load_data()
        data.set_target_column(self.target_column())
        data.metadata = self.metadata

        feature_columns = self.render_tf_feature_columns(data=data)
        data.set_tf_feature_columns(feature_columns)

        split_data = self.split_validation_and_test_data(data=data)
        ai.set_evaluation_data(split_data['validation_data'])
        ai.set_training_data(split_data['training_data'])

    def load_data(self) -> DataModel:
        loader = DataLoader()

        self.load_file(loader)

        columns = self.get_feature_column_names()
        columns.append(self.target_column())
        columns = columns + self.data_columns
        loader.filter_columns(columns)

        return loader.get_dataset()

    def load_file(self, loader: DataLoader):
        if 'csv' in self.data_source():
            loader.load_csv(self.data_source())
            return

        raise RuntimeError('Failed to load data from {}.'.format(self.data_source()))

    def get_feature_column_names(self) -> list:
        names = []
        for feature_column in self.feature_columns():
            names.append(feature_column['name'])

        return names

    def split_validation_and_test_data(self, data: DataModel):
        splitter = DataSetSplitter(data_model=data)
        result = splitter.split_by_ratio([self.training_data_percentage(), self.validation_data_percentage()])

        return {'training_data': result[0], 'validation_data': result[1]}

    # todo: possible separate builder, refactor creating of feature columns to sepperate builder.
    #  Feature columns must be rendered after scrubbing because the columns might have changed and
    #  catergoric columns should have no None values in them.
    def render_tf_feature_columns(self, data: DataModel) -> list:
        tf_feature_columns = []
        for feature_column in self.feature_columns():
            column = None
            if feature_column['type'] is self.CATEGORICAL_COLUMN_VOC_LIST:
                column = self.build_categorical_column_voc_list(feature_column, data)
            elif feature_column['type'] is self.NUMERICAL_COLUMN:
                column = self.build_numerical_column(feature_column['name'])

            if column is None:
                raise RuntimeError('feature column not set, ({})'.format(feature_column))

            tf_feature_columns.append(column)

        return tf_feature_columns

    @staticmethod
    def build_numerical_column(feature_column: dict) -> tf.feature_column.numeric_column:
        return tf.feature_column.numeric_column(feature_column)

    @staticmethod
    def build_categorical_column_voc_list(
            feature_column_data: dict,
            data: DataModel
    ) -> tf.feature_column.categorical_column_with_vocabulary_list:

        categories = data.get_all_column_categories(feature_column_data['name'])

        # todo: refactor so tf columns are manufactured in different builder after scrubbing.
        #  this one too.
        filtered_categories = [cat for cat in categories if type(cat) is str]

        return tf.feature_column.categorical_column_with_vocabulary_list(
            feature_column_data['name'],
            vocabulary_list=filtered_categories
        )


class TestDataBuilder(unittest.TestCase):

    def test_build(self):
        data_builder = DataBuilder(data_source='../../../data/test_data.csv',
                                   target_column='target_1',
                                   validation_data_percentage=20,
                                   feature_columns={},
                                   metadata=MetaData())

        data_builder.add_feature_column(name='feature_1', column_type=DataBuilder.CATEGORICAL_COLUMN_VOC_LIST)
        data_builder.add_feature_column(name='feature_2', column_type=DataBuilder.NUMERICAL_COLUMN)
        data_builder.add_feature_column(name='feature_3', column_type=DataBuilder.NUMERICAL_COLUMN)

        # todo: create factory
        arti = AI()
        data_builder.validate()
        data_builder.build(ai=arti)

        feature_names = ['feature_1', 'feature_2', 'feature_3']
        self.validate_data_frame(arti.training_data, feature_names)
        self.validate_data_frame(arti.evaluation_data, feature_names)

    def test_constructor_build(self):
        data_builder = DataBuilder(data_source='../../../data/test_data.csv',
                                   target_column='target_1',
                                   validation_data_percentage=20,
                                   feature_columns={
                                        'feature_1': DataBuilder.CATEGORICAL_COLUMN_VOC_LIST,
                                        'feature_2': DataBuilder.NUMERICAL_COLUMN,
                                        'feature_3': DataBuilder.NUMERICAL_COLUMN
                                   },
                                   metadata=MetaData())

        arti = AI()
        data_builder.validate()
        data_builder.build(ai=arti)

        feature_names = ['feature_1', 'feature_2', 'feature_3']
        self.validate_data_frame(arti.training_data, feature_names)
        self.validate_data_frame(arti.evaluation_data, feature_names)

    def validate_data_frame(self, data_frame: DataModel, feature_name_list: list):
        self.assertEqual(data_frame.feature_columns_names, feature_name_list)
        self.assertEqual(data_frame.target_column_name, 'target_1')

        for tf_feature_column in data_frame.get_tf_feature_columns():
            self.assertTrue(tf_feature_column.name in feature_name_list)


if __name__ == '__main__':
    unittest.main()
