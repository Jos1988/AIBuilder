import unittest
from abc import ABC, abstractmethod


class AIRecipe:
    def __init__(self, ingredients: dict):
        self.ingredients = ingredients

    def get_ingredients(self) -> dict:
        return self.ingredients

    def get_ingredient_types(self):
        types = []
        for ingredient_type, specification in self.ingredients.items():
            types.append(ingredient_type)

        return types

    def add_ingredient(self, ingredient_type: str, specification):
        self.ingredients[ingredient_type] = specification

    def get_ingredient_specification(self, ingredient_type: str):
        if ingredient_type in self.ingredients:
            return self.ingredients[ingredient_type]

        return None


class TestIngredient(unittest.TestCase):

    def setUp(self):
        self.recipe_description = {
            'spices': ['salt', 'pepper'],
            'filling': 'potatoes',
            'meat': 'steak',
            'sauce': 'brown game stock'
        }

        self.recipe = AIRecipe(self.recipe_description)

    def test_get_ingredients_type(self):
        expected = ['spices', 'meat', 'sauce', 'filling']
        types = self.recipe.get_ingredient_types()

        self.assertCountEqual(expected, types, 'Expected ingredient types do not match.')

    def test_get_ingredients(self):
        self.assertEqual(self.recipe_description, self.recipe.get_ingredients())

    def test_add_ingredient(self):
        self.recipe.add_ingredient('garnish', 'parsley')
        specification = self.recipe.ingredients['garnish']

        self.assertEqual('parsley', specification)

    def test_get_ingredient_specification(self):
        specification = self.recipe.get_ingredient_specification('spices')

        self.assertEqual(self.recipe_description['spices'], specification)


class RecipeCollection(ABC):

    @property
    @abstractmethod
    def recipes(self):
        pass

    @abstractmethod
    def load_recipes(self):
        pass

    def get_recipes(self):
        return self.recipes

    @abstractmethod
    def validate_recipes(self):
        pass


if __name__ == '__main__':
    unittest.main()