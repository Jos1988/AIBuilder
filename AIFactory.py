from AIBuilder.AIRecipe import AIRecipe
from AIBuilder.AI import AI
from abc import ABC, abstractmethod
import tensorflow as tf
import unittest


class Builder(ABC):

    @abstractmethod
    def valid(self) -> bool:
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
        return 'data'

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
    # format:
    # 'estimator':
    # 'type' : 'linear_regressor'

    ESTIMATOR_TYPE = 'type'
    LINEAR_REGRESSOR_TYPE = 'linear_regressor'

    @property
    def required_specifications(self) -> dict:
        return {self.ESTIMATOR_TYPE: [self.LINEAR_REGRESSOR_TYPE]}

    @property
    def optional_specifications(self) -> dict:
        return {}

    @property
    def dependent_on(self) -> list:
        return [self.OPTIMIZER, self.DATA_MODEL]

    @property
    def ingredient_type(self) -> str:
        return self.ESTIMATOR

    def build(self, neural_net: AI, recipe: AIRecipe):
        specifications = recipe.get_ingredient_specification(self.ingredient_type)
        regressor_type = specifications[self.ESTIMATOR_TYPE]

        if regressor_type is self.LINEAR_REGRESSOR_TYPE:
            regressor = tf.estimator.LinearRegressor(
                feature_columns=neural_net.training_data.get_tf_feature_columns(),
                optimizer=neural_net.optimizer
            )

            neural_net.set_estimator(regressor)
            return

        raise RuntimeError('Estimator Builder failed to set estimator.')


class TestEstimatorBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = EstimatorBuilder()

    def test_validate(self):
        recipe = AIRecipe({self.builder.ESTIMATOR: {
            self.builder.ESTIMATOR_TYPE: self.builder.LINEAR_REGRESSOR_TYPE
        }})

        self.builder.validate(recipe)

    def test_invalid_specifications(self):
        recipe = AIRecipe({self.builder.ESTIMATOR: {
            self.builder.ESTIMATOR_TYPE: 'invalid'
        }})

        with self.assertRaises(RuntimeError):
            self.builder.validate(recipe)

    def test_missing_specifications(self):
        recipe = AIRecipe({'invalid': {
            self.builder.ESTIMATOR_TYPE: self.builder.LINEAR_REGRESSOR_TYPE
        }})

        with self.assertRaises(RuntimeError):
            self.builder.validate(recipe)


class OptimizerBuilder(Builder):
    # format:
    # 'optimizer' :
    # 'type': 'gradient_descent_optimizer'
    # 'learning_rate' : float
    # (optional)'gradient_clipping' : float

    LEARNING_RATE = 'learning_rate'
    GRADIENT_CLIPPING = 'gradient_clipping'

    OPTIMIZER_TYPE = 'type'
    GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent_optimizer'

    @property
    def required_specifications(self) -> dict:
        return {self.LEARNING_RATE: float,
                self.OPTIMIZER_TYPE: [self.GRADIENT_DESCENT_OPTIMIZER]}

    @property
    def optional_specifications(self) -> dict:
        return {self.GRADIENT_CLIPPING: float}

    @property
    def dependent_on(self) -> list:
        return []

    @property
    def ingredient_type(self) -> str:
        return self.OPTIMIZER

    def build(self, neural_net: AI, recipe: AIRecipe):
        specification = recipe.get_ingredient_specification(self.ingredient_type)
        learning_rate = specification[self.LEARNING_RATE]
        optimizer_type = specification[self.OPTIMIZER_TYPE]
        clipping = None
        if specification[self.GRADIENT_CLIPPING] is not None:
            clipping = specification[self.GRADIENT_CLIPPING]

        my_optimizer = self._set_optimizer(optimizer_type=optimizer_type, learning_rate=learning_rate)

        if clipping is not None:
            my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, clipping)

        neural_net.set_optimizer(my_optimizer)

    def _set_optimizer(self, optimizer_type: str, learning_rate: int):
        if optimizer_type is self.GRADIENT_DESCENT_OPTIMIZER:
            my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            return my_optimizer

        raise RuntimeError('Optimizer not set.')


class TestOptimizerBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = OptimizerBuilder()

    def test_validate(self):
        recipe_with_clipping = AIRecipe({self.builder.OPTIMIZER: {
            self.builder.OPTIMIZER_TYPE: self.builder.GRADIENT_DESCENT_OPTIMIZER,
            self.builder.GRADIENT_CLIPPING: 5.0,
            self.builder.LEARNING_RATE: 0.0002
        }})

        recipe_no_clipping = AIRecipe({self.builder.OPTIMIZER: {
            self.builder.OPTIMIZER_TYPE: self.builder.GRADIENT_DESCENT_OPTIMIZER,
            self.builder.LEARNING_RATE: 0.0002
        }})

        self.builder.validate(recipe_with_clipping)
        self.builder.validate(recipe_no_clipping)

    def test_invalid_specifications(self):
        recipe_invalid_type = AIRecipe({self.builder.ESTIMATOR: {
            self.builder.OPTIMIZER_TYPE: 'invalid',
            self.builder.GRADIENT_CLIPPING: 5.0,
            self.builder.LEARNING_RATE: 0.0002
        }})

        recipe_invalid_clipping = AIRecipe({self.builder.ESTIMATOR: {
            self.builder.OPTIMIZER_TYPE: self.builder.GRADIENT_DESCENT_OPTIMIZER,
            self.builder.GRADIENT_CLIPPING: 'invalid',
            self.builder.LEARNING_RATE: 0.0002
        }})

        recipe_invalid_learning_rate = AIRecipe({self.builder.ESTIMATOR: {
            self.builder.OPTIMIZER_TYPE: self.builder.GRADIENT_DESCENT_OPTIMIZER,
            self.builder.GRADIENT_CLIPPING: 5.0,
            self.builder.LEARNING_RATE: 1
        }})

        with self.assertRaises(RuntimeError):
            self.builder.validate(recipe_invalid_type)

        with self.assertRaises(RuntimeError):
            self.builder.validate(recipe_invalid_clipping)

        with self.assertRaises(RuntimeError):
            self.builder.validate(recipe_invalid_learning_rate)

    def test_missing_specifications(self):
        recipe = AIRecipe({'invalid': {
            self.builder.OPTIMIZER_TYPE: self.builder.GRADIENT_DESCENT_OPTIMIZER
        }})

        with self.assertRaises(RuntimeError):
            self.builder.validate(recipe)


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
