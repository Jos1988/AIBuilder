from AIBuilder.AIRecipe import AIRecipe
from AIBuilder.AI import AI
from abc import ABC, abstractmethod
import tensorflow as tf
import unittest


class Builder(ABC):
    # register you new builder_type here.
    ESTIMATOR = 'estimator'
    OPTIMIZER = 'optimizer'

    @property
    @abstractmethod
    def required_specifications(self) -> dict:
        pass

    @property
    @abstractmethod
    def optional_specifications(self) -> dict:
        pass

    @property
    @abstractmethod
    def dependent_on(self) -> list:
        pass

    @property
    @abstractmethod
    def ingredient_type(self) -> str:
        pass

    def accepts(self, ingredient_type: str) -> bool:
        return self.ingredient_type is ingredient_type

    def validate(self, recipe: AIRecipe):
        specifications = recipe.get_ingredient_specification(self.ingredient_type)

        if specifications is None:
            raise RuntimeError('Specifications for {} missing.'.format(__class__))

        self._load_required_specifications(specifications=specifications)
        self._load_optional_specification(specifications=specifications)

    def _load_optional_specification(self, specifications: dict):
        for optional_specification, specification_format in self.optional_specifications.items():
            if optional_specification not in specifications:
                continue

            value = specifications[optional_specification]
            self._validate_specification_format(value, optional_specification, specification_format)

    def _load_required_specifications(self, specifications: dict):
        for required_specification, specification_format in self.required_specifications.items():
            if required_specification not in specifications:
                raise RuntimeError('Required specification {} is missing.'.format(required_specification))

            value = specifications[required_specification]
            self._validate_specification_format(value, required_specification, specification_format)

    def _validate_specification_format(self, value, specification: str, specification_format):
            if isinstance(specification_format, list):
                if value in specification_format:
                    return

                raise RuntimeError('unknown value ({}) passed to {} specification in {} of {} builder'
                                   .format(value, specification, self.ingredient_type, self.ingredient_type))

            if value is None or type(value) is not specification_format:
                raise RuntimeError('{} is missing specification value is {} in stead of {}.'
                                   .format(self.ingredient_type, type(value), specification_format))

    @abstractmethod
    def build(self, neural_net: AI, recipe: AIRecipe):
        pass


class EstimatorBuilder(Builder):
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
        return [self.OPTIMIZER]

    @property
    def ingredient_type(self) -> str:
        return self.ESTIMATOR

    def build(self, neural_net: AI, recipe: AIRecipe):
        specifications = recipe.get_ingredient_specification(self.ingredient_type)
        regressor_type = specifications[self.ESTIMATOR]

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
        # load Decorators
        # self.AIDecorators.append(AIDecorator())
        # ...

    def create_AI(self, recipe: AIRecipe) -> AI:
        artificial_intelligence = AI()
        required_builders = []

        ingredient_types = recipe.get_ingredient_types()

        for ingredient_type in ingredient_types:
            builder = self.get_builder(ingredient_type)
            required_builders.append(builder)

        for builder in required_builders:
            builder.validate(recipe)

        # sort builders

        for builder in required_builders:
            builder.build(artificial_intelligence, recipe)

        return artificial_intelligence

    def get_builder(self, ingredient_type: str) -> Builder:
        valid_decorators = []
        for decorator in self.AIBuilders:
            if decorator.accepts(ingredient_type):
                valid_decorators.append(decorator)

        if valid_decorators.count() is 1:
            return valid_decorators.pop()

        raise RuntimeError('{} decorators found for ingredient: {}'.format(valid_decorators.count(), ingredient_type))


if __name__ == '__main__':
    unittest.main()
