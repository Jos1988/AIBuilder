from AIBuilder.AIRecipe import AIRecipe
from AIBuilder.AI import AI
from abc import ABC, abstractmethod
import tensorflow as tf
import unittest
from typing import Optional, Union


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
    def validate(self) -> bool:
        pass

    @abstractmethod
    def build(self, neural_net: AI, recipe: AIRecipe):
        pass


class DataBuilder(Builder, ABC):
    """
     format
    'data': {
      'data_source: string
      'feature_columns': [{name: string, 'type': 'categorical_column_with_vocabulary_list'}]
      target_column: string
    }
    """

    @property
    def required_specifications(self) -> dict:
        return {'data_source': str, 'feature_columns': [{'name': str, 'type': str}], 'target_column': str}

    @property
    def optional_specifications(self) -> dict:
        return {}

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def ingredient_type(self) -> str:
        return self.DATA_MODEL

    def build(self, neural_net: AI, recipe: AIRecipe):
        pass

    # Load data.
    # loader = Data.DataLoader()
    # loader.load_csv('data/7210_1.csv')
    # loader.filter_columns(
    #     [
    #         'brand',
    #         'manufacturer',
    #         'categories',
    #         'colors',
    #         'name',
    #         'prices.amountMin',
    #         'prices.amountMax',
    #         'prices.currency',
    #         'prices.dateSeen'
    #     ]
    # )
    #
    # data_model = loader.get_dataset()
    #
    # ml_dataset.set_target_column('prices.target')
    # ml_dataset.set_tf_feature_columns([
    #     tf.feature_column.categorical_column_with_vocabulary_list(
    #         'brand',
    #         vocabulary_list=ml_dataset.get_all_column_categories('brand')
    #     ),
    #     tf.feature_column.categorical_column_with_vocabulary_list(
    #         'manufacturer',
    #         vocabulary_list=ml_dataset.get_all_column_categories('manufacturer')
    #     )
    # ])


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
