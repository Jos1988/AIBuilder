from AIBuilder.AIRecipe import AIRecipe
from AIBuilder.AI import AI
from AIBuilder.Data import DataModel, DataSetSplitter, DataLoader
from abc import ABC, abstractmethod
import tensorflow as tf
import unittest
from typing import Optional


class Builder(ABC):
    # register you new builder_type here.
    ESTIMATOR = 'estimator'
    OPTIMIZER = 'optimizer'
    DATA_MODEL = 'data_model'

    @property
    @abstractmethod
    def dependent_on(self) -> list:
        pass

    @property
    @abstractmethod
    def ingredient_type(self) -> str:
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def build(self, neural_net: AI):
        pass


class DataBuilder(Builder):
    CATEGORICAL_COLUMN_VOC_LIST = 'categorical_column_with_vocabulary_list'
    NUMERICAL_COLUMN = 'numeric_column'
    valid_column_types = [CATEGORICAL_COLUMN_VOC_LIST, NUMERICAL_COLUMN]

    def __init__(self, data_source: str, target_column: str, validation_data_percentage: int):
        self.validation_data_percentage = validation_data_percentage
        self.training_data_percentage = 100 - self.validation_data_percentage
        self.data_source = data_source
        self.target_column = target_column
        self.feature_columns = []
        self.test_data = None
        self.validation_data = None

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def ingredient_type(self) -> str:
        return self.DATA_MODEL

    def add_feature_column(self, name: str, column_type: str):
        self.is_valid_column_type(column_type)
        self.feature_columns.append({'name': name, 'type': column_type})

    def validate(self):
        assert type(self.validation_data_percentage) is int, 'Invalid validation data percentage, {}.'.format(
            self.validation_data_percentage)

        assert 0 < self.validation_data_percentage < 100, 'validation data perc. must be between 0 and 100, {} given'.format(
            self.validation_data_percentage)

        assert type(self.data_source) is str, 'Invalid data source for data model, {}.'.format(self.data_source)

        assert type(self.target_column) is str, 'Invalid target column name set, {}'.format(self.target_column)

        assert self.target_column not in self.get_feature_column_names(), \
            'target column {}, also set as feature column!'.format(self.target_column)

        assert type(self.feature_columns) is list, 'Invalid list of feature columns set, {}.'.format(
            self.feature_columns)

        assert len(self.feature_columns) is not 0, 'No feature columns provided!'

        for feature_column in self.feature_columns:
            self.is_valid_column_type(feature_column['type'])
            assert type(feature_column['name']) is str, 'Invalid feature column name, {}'.format(feature_column)

        for feature_column in self.feature_columns:
            self.is_valid_column_type(feature_column['type'])

    def build(self, ai: AI):
        data = self.load_data()
        data.set_target_column(self.target_column)

        feature_columns = self.render_tf_feature_columns(data=data)
        data.set_tf_feature_columns(feature_columns)

        split_data = self.split_validation_and_test_data(data=data)
        ai.set_validation_data(split_data['validation_data'])
        ai.set_training_data(split_data['training_data'])

    def load_data(self) -> DataModel:
        loader = DataLoader()

        self.load_file(loader)

        columns = self.get_feature_column_names()
        columns.append(self.target_column)

        loader.filter_columns(columns)

        return loader.get_dataset()

    def load_file(self, loader: DataLoader):
        if 'csv' in self.data_source:
            loader.load_csv(self.data_source)
            return

        raise RuntimeError('Failed to load data from {}.'.format(self.data_source))

    def get_feature_column_names(self) -> list:
        names = []
        for feature_column in self.feature_columns:
            names.append(feature_column['name'])

        return names

    def split_validation_and_test_data(self, data: DataModel):
        splitter = DataSetSplitter(data_model=data)
        result = splitter.split_by_ratio([self.training_data_percentage, self.validation_data_percentage])

        return {'training_data': result[0], 'validation_data': result[1]}

    def is_valid_column_type(self, column_type: str):
        assert column_type in self.valid_column_types, 'feature column type ({}) not valid, should be in {}'.format(
            column_type, self.valid_column_types)

    def render_tf_feature_columns(self, data: DataModel) -> list:
        tf_feature_columns = []
        for feature_column in self.feature_columns:
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

        return tf.feature_column.categorical_column_with_vocabulary_list(
            feature_column_data['name'],
            vocabulary_list=data.get_all_column_categories(feature_column_data['name'])
        )


class TestDataBuilder(unittest.TestCase):

    def test_build(self):
        data_builder = DataBuilder(data_source='../data/test_data.csv',
                                   target_column='target_1',
                                   validation_data_percentage=20)

        data_builder.add_feature_column(name='feature_1', column_type=DataBuilder.CATEGORICAL_COLUMN_VOC_LIST)
        data_builder.add_feature_column(name='feature_2', column_type=DataBuilder.NUMERICAL_COLUMN)
        data_builder.add_feature_column(name='feature_3', column_type=DataBuilder.NUMERICAL_COLUMN)

        arti = AI()
        data_builder.validate()
        data_builder.build(ai=arti)

        feature_names = ['feature_1', 'feature_2', 'feature_3']
        self.validate_data_frame(arti.training_data, feature_names)
        self.validate_data_frame(arti.validation_data, feature_names)

    def validate_data_frame(self, data_frame: DataModel, feature_name_list: list):
        self.assertEqual(data_frame.feature_columns_names, feature_name_list)
        self.assertEqual(data_frame.target_column_name, 'target_1')

        for tf_feature_column in data_frame.get_tf_feature_columns():
            self.assertTrue(tf_feature_column.name in feature_name_list)


class EstimatorBuilder(Builder):
    LINEAR_REGRESSOR = 'linear_regressor'
    valid_estimator_types = [LINEAR_REGRESSOR]
    estimator: str

    def __init__(self, estimator_type: str):
        self.estimator_type = None
        self.set_estimator(estimator_type)

    @property
    def dependent_on(self) -> list:
        return [self.OPTIMIZER, self.DATA_MODEL]

    @property
    def ingredient_type(self) -> str:
        return self.ESTIMATOR

    def set_estimator(self, estimator_type):
        self.validate_estimator(estimator_type)
        self.estimator_type = estimator_type

    def validate_estimator(self, estimator_type: str):
        assert estimator_type in self.valid_estimator_types, 'Unknown type of estimator {}, must be in {}'.format(
            estimator_type, self.valid_estimator_types)

    def validate(self) -> bool:
        self.validate_estimator(self.estimator_type)
        return True

    def build(self, neural_net: AI, recipe: AIRecipe):
        if self.estimator_type is self.LINEAR_REGRESSOR:
            estimator = tf.estimator.LinearRegressor(
                feature_columns=neural_net.training_data.get_tf_feature_columns(),
                optimizer=neural_net.optimizer
            )

            neural_net.set_estimator(estimator)
            return

        raise RuntimeError('Estimator Builder failed to set estimator.')


class TestEstimatorBuilder(unittest.TestCase):

    def test_validate(self):
        estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)
        estimator_builder.validate()

    def test_invalid_estimator_type(self):
        invalid_estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)
        invalid_estimator_builder.estimator_type = 'invalid'

        valid_estimator_builder = EstimatorBuilder(EstimatorBuilder.LINEAR_REGRESSOR)

        with self.assertRaises(AssertionError):
            invalid_estimator_builder.validate()

        with self.assertRaises(AssertionError):
            valid_estimator_builder.set_estimator('invalid')

        with self.assertRaises(AssertionError):
            EstimatorBuilder('invalid')

    def test_build(self):
        pass


class OptimizerBuilder(Builder):
    LEARNING_RATE = 'learning_rate'
    GRADIENT_CLIPPING = 'gradient_clipping'

    GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent_optimizer'

    valid_optimizer_types = [GRADIENT_DESCENT_OPTIMIZER]

    def __init__(self, optimizer_type: str, learning_rate: float, gradient_clipping: Optional[float] = None):
        self.validate_optimizer_type(optimizer_type)
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def ingredient_type(self) -> str:
        return self.OPTIMIZER

    def validate(self) -> bool:
        assert self.learning_rate is not float, 'optimizer learning rate must be float, currently: {}'.format(
            self.learning_rate)

        self.validate_optimizer_type(self.optimizer_type)

        assert type(self.gradient_clipping) is float or self.gradient_clipping is None, \
            'gradient clipping must be float or None, currently {}'.format(self.gradient_clipping)

    def build(self, neural_net: AI, recipe: AIRecipe):
        my_optimizer = self._set_optimizer(optimizer_type=self.optimizer_type, learning_rate=self.learning_rate)

        if self.gradient_clipping is not None:
            my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, self.gradient_clipping)

        neural_net.set_optimizer(my_optimizer)

    def _set_optimizer(self, optimizer_type: str, learning_rate: float) -> tf.train.Optimizer:
        if optimizer_type is self.GRADIENT_DESCENT_OPTIMIZER:
            my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            return my_optimizer

        raise RuntimeError('Optimizer not set.')

    def validate_optimizer_type(self, optimizer_type: str):
        assert optimizer_type in self.valid_optimizer_types, 'Unknown type op optimizer {}, must be in {}'.format(
            optimizer_type, self.valid_optimizer_types)


class TestOptimizerBuilder(unittest.TestCase):

    def test_valid_validate(self):
        optimizer_builder_no_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0)

        optimizer_builder_with_clipping = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=1.0,
            gradient_clipping=1.0)

        optimizer_builder_no_clipping.validate()
        optimizer_builder_with_clipping.validate()

    def test_invalid_validate(self):
        with self.assertRaises(AssertionError):
            OptimizerBuilder(
                optimizer_type='invalid',
                learning_rate=5.0,
                gradient_clipping=0.0002)

        optimizer_builder = OptimizerBuilder(
            optimizer_type=OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER,
            learning_rate=0.1,
            gradient_clipping=0.0002)

        optimizer_builder.optimizer_type = 'invalid'

        with self.assertRaises(AssertionError):
            optimizer_builder.validate()

    def test_build(self):
        pass


class AIFactory:
    def __init__(self):
        self.AIBuilders = []
        # load you Builders here instead of using DI
        self.AIBuilders.append(EstimatorBuilder())
        self.AIBuilders.append(OptimizerBuilder())

        self.required_builders = []

    def create_AI(self, recipe: AIRecipe) -> AI:
        artificial_intelligence = AI()

        ingredient_types = recipe.get_ingredient_types()

        for ingredient_type in ingredient_types:
            builder = self.get_builder(ingredient_type)
            self.required_builders.append(builder)

        for builder in self.required_builders:
            builder.validate(recipe)

        # self.sortBuilders()

        for builder in self.required_builders:
            builder.build(artificial_intelligence, recipe)

        return artificial_intelligence

    def get_builder(self, ingredient_type: str) -> Builder:
        valid_decorators = []
        for decorator in self.AIBuilders:
            if decorator.accepts(ingredient_type):
                valid_decorators.append(decorator)

        if len(valid_decorators) is 1:
            return valid_decorators.pop()

        raise RuntimeError('{} decorators found for ingredient: {}'.format(len(valid_decorators), ingredient_type))

    def sortBuilders(self, builders: list):
        pass


class TestAIFactory(unittest.TestCase):

    def setUp(self):
        self.factory = AIFactory()

    # def test_create_AI(self):
    #     # 'type': 'gradient_descent_optimizer'
    #     # 'learning_rate' : float
    #     # (optional)'gradient_clipping' : float
    #
    #     # 'estimator':
    #     # 'type' : 'linear_regressor'
    #
    #     # todo add datamodel builder as optimizerbuilder depends on it.
    #     recipe = AIRecipe({
    #         'estimator': {'type': 'linear_regressor'},
    #         'optimizer': {'type': 'gradient_descent_optimizer', 'learning_rate': 0.0002, 'gradient_clipping': 5.0}
    #     })
    #
    #     artie = self.factory.create_AI(recipe=recipe)
    #     print(type(artie.optimizer))
    #     print(type(artie.estimator))
    #     print(artie)


if __name__ == '__main__':
    unittest.main()
