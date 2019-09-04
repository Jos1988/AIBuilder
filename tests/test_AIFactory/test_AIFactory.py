from AIBuilder.AIFactory.AIFactory import CachingInstructionsLoader, BuilderInstructionModel
from AIBuilder.AIFactory.Builders import Builder
from AIBuilder.AIFactory.smartCache.SmartCache import SmartCacheManager, InstructionsRepository, CallCountLog
from AIBuilder.Data import MetaData
from typing import List
import unittest
from unittest import mock, TestCase
from AIBuilder.AIFactory import AIFactory


class TestAIFactory(TestCase):

    def setUp(self):
        self.factory = AIFactory.AIFactory(builders=[], project_name='test', log_dir='test/dir')

        metadata = MetaData()
        metadata.define_numerical_columns(['feature_2', 'feature_3', 'target_1'])
        metadata.define_categorical_columns(['feature_1'])

        self.builder_A = mock.Mock(spec=Builder)
        self.builder_A.dependent_on = ['B']
        self.builder_A.builder_type = 'A'
        self.builder_A.describe = mock.Mock()
        self.builder_A.describe.return_value = {'A': 1}
        self.builder_A.build = mock.Mock()

        self.builder_B = mock.Mock(spec=Builder)
        self.builder_B.dependent_on = ['C']
        self.builder_B.builder_type = 'B'
        self.builder_B.describe = mock.Mock()
        self.builder_B.describe.return_value = {'B': 1}
        self.builder_B.build = mock.Mock()

        self.builder_C = mock.Mock(spec=Builder)
        self.builder_C.dependent_on = []
        self.builder_C.builder_type = 'C'
        self.builder_C.describe = mock.Mock()
        self.builder_C.describe.return_value = {'C': 1}
        self.builder_C.build = mock.Mock()

        self.artie = self.factory.create_AI([self.builder_A, self.builder_B, self.builder_C])

    def test_create_AI(self):
        self.assertEqual(self.artie.description, {'C': {'C': 1}, 'B': {'B': 1}, 'A': {'A': 1}})

        self.builder_A.build.assert_called_once_with(self.artie)
        self.builder_B.build.assert_called_once_with(self.artie)
        self.builder_C.build.assert_called_once_with(self.artie)


class TestBuilderSorter(unittest.TestCase):

    def setUp(self):
        self.builder_A = mock.Mock(spec=Builder)
        self.builder_A.dependent_on = ['B']
        self.builder_A.builder_type = 'A'
        self.builder_A.describe = mock.Mock()
        self.builder_A.describe.return_value = {'A': 1}
        self.builder_A.build = mock.Mock()

        self.builder_B = mock.Mock(spec=Builder)
        self.builder_B.dependent_on = ['C']
        self.builder_B.builder_type = 'B'
        self.builder_B.describe = mock.Mock()
        self.builder_B.describe.return_value = {'B': 1}
        self.builder_B.build = mock.Mock()

        self.builder_C = mock.Mock(spec=Builder)
        self.builder_C.dependent_on = []
        self.builder_C.builder_type = 'C'
        self.builder_C.describe = mock.Mock()
        self.builder_C.describe.return_value = {'C': 1}
        self.builder_C.build = mock.Mock()

        self.sorter = AIFactory.BuilderSorter()

    def test_sort(self):
        sorter_builders = self.sorter.sort([self.builder_A, self.builder_B, self.builder_C])

        self.assertListEqual(sorter_builders, [self.builder_C, self.builder_B, self.builder_A])


class TestPermutationGenerator(unittest.TestCase):

    def setUp(self):
        self.builder_A1 = mock.Mock(name='A1', spec=Builder)
        self.builder_A1.dependent_on = ['B']
        self.builder_A1.builder_type = 'A'
        self.builder_A1.describe = mock.Mock()
        self.builder_A1.describe.return_value = {'A': 1}
        self.builder_A1.build = mock.Mock()

        self.builder_A2 = mock.Mock(name='A2', spec=Builder)
        self.builder_A2.dependent_on = ['B']
        self.builder_A2.builder_type = 'A'
        self.builder_A2.describe = mock.Mock()
        self.builder_A2.describe.return_value = {'A': 2}
        self.builder_A2.build = mock.Mock()

        self.builder_B1 = mock.Mock(name='B1', spec=Builder)
        self.builder_B1.dependent_on = ['C']
        self.builder_B1.builder_type = 'B'
        self.builder_B1.describe = mock.Mock()
        self.builder_B1.describe.return_value = {'B': 1}
        self.builder_B1.build = mock.Mock()

        self.builder_B2 = mock.Mock(name='B2', spec=Builder)
        self.builder_B2.dependent_on = ['C']
        self.builder_B2.builder_type = 'B'
        self.builder_B2.describe = mock.Mock()
        self.builder_B2.describe.return_value = {'B': 2}
        self.builder_B2.build = mock.Mock()

        self.builder_C = mock.Mock(name='C', spec=Builder)
        self.builder_C.dependent_on = ['D']
        self.builder_C.builder_type = 'C'
        self.builder_C.describe = mock.Mock()
        self.builder_C.describe.return_value = {'C': 1}
        self.builder_C.build = mock.Mock()

        self.builder_D1 = mock.Mock(name='D1', spec=Builder)
        self.builder_D1.dependent_on = []
        self.builder_D1.builder_type = 'D'
        self.builder_D1.describe = mock.Mock()
        self.builder_D1.describe.return_value = {'D': 1}
        self.builder_D1.build = mock.Mock()

        self.builder_D2 = mock.Mock(name='D2', spec=Builder)
        self.builder_D2.dependent_on = []
        self.builder_D2.builder_type = 'D'
        self.builder_D2.describe = mock.Mock()
        self.builder_D2.describe.return_value = {'D': 2}
        self.builder_D2.build = mock.Mock()

        self.permutation_generator = AIFactory.PermutationGenerator()

    def test_permute(self):
        builders = [self.builder_A1, self.builder_A2,
                    self.builder_B1, self.builder_B2,
                    self.builder_C,
                    self.builder_D1, self.builder_D2]

        permutation1 = [self.builder_A1, self.builder_B1, self.builder_C, self.builder_D1]
        permutation2 = [self.builder_A1, self.builder_B2, self.builder_C, self.builder_D1]
        permutation3 = [self.builder_A2, self.builder_B1, self.builder_C, self.builder_D1]
        permutation4 = [self.builder_A2, self.builder_B2, self.builder_C, self.builder_D1]
        permutation5 = [self.builder_A1, self.builder_B1, self.builder_C, self.builder_D2]
        permutation6 = [self.builder_A1, self.builder_B2, self.builder_C, self.builder_D2]
        permutation7 = [self.builder_A2, self.builder_B1, self.builder_C, self.builder_D2]
        permutation8 = [self.builder_A2, self.builder_B2, self.builder_C, self.builder_D2]
        expected_permutations = [permutation1, permutation2, permutation3, permutation4, permutation5, permutation6,
                                 permutation7, permutation8]

        resulting_permutations = self.permutation_generator.generate(builders=builders)

        seen_permutations = []
        for resulting_permutation in resulting_permutations:
            if resulting_permutation in seen_permutations:
                raise AssertionError('permutation found twice in {}'.format(resulting_permutations))

            if resulting_permutation in expected_permutations:
                seen_permutations.append(resulting_permutation)

        self.assertTrue(len(seen_permutations), len(expected_permutations))


class TestCachingInstructionsLoaderComplex(unittest.TestCase):

    def setUp(self) -> None:
        self.builder_A = mock.Mock(name='A', spec=Builder)
        self.builder_A.dependent_on = ['B']
        self.builder_A.builder_type = 'A'
        self.builder_A.__hash__ = mock.Mock()
        self.builder_A.__hash__.return_value = 'A'
        self.builder_A.build = mock.Mock()

        self.builder_B1 = mock.Mock(name='B1', spec=Builder)
        self.builder_B1.dependent_on = ['C']
        self.builder_B1.builder_type = 'B'
        self.builder_B1.__hash__ = mock.Mock()
        self.builder_B1.__hash__.return_value = 'B1'
        self.builder_B1.build = mock.Mock()

        self.builder_B2 = mock.Mock(name='B2', spec=Builder)
        self.builder_B2.dependent_on = ['C']
        self.builder_B2.builder_type = 'B'
        self.builder_B2.__hash__ = mock.Mock()
        self.builder_B2.__hash__.return_value = 'B2'
        self.builder_B2.build = mock.Mock()

        self.builder_C = mock.Mock(name='C', spec=Builder)
        self.builder_C.dependent_on = ['D']
        self.builder_C.builder_type = 'C'
        self.builder_C.__hash__ = mock.Mock()
        self.builder_C.__hash__.return_value = 'C'
        self.builder_C.build = mock.Mock()

        self.builder_D1 = mock.Mock(name='D1', spec=Builder)
        self.builder_D1.dependent_on = []
        self.builder_D1.builder_type = 'D'
        self.builder_D1.__hash__ = mock.Mock()
        self.builder_D1.__hash__.return_value = 'D1'
        self.builder_D1.build = mock.Mock()

        self.builder_D2 = mock.Mock(name='D2', spec=Builder)
        self.builder_D2.dependent_on = ['E']
        self.builder_D2.builder_type = 'D'
        self.builder_D2.__hash__ = mock.Mock()
        self.builder_D2.__hash__.return_value = 'D2'
        self.builder_D2.build = mock.Mock()

        self.builder_E = mock.Mock(name='E', spec=Builder)
        self.builder_E.dependent_on = []
        self.builder_E.builder_type = 'E'
        self.builder_E.__hash__ = mock.Mock()
        self.builder_E.__hash__.return_value = 'E'
        self.builder_E.build = mock.Mock()

        permutation1 = [self.builder_A, self.builder_B1, self.builder_C, self.builder_D1, self.builder_E]
        permutation2 = [self.builder_A, self.builder_B1, self.builder_C, self.builder_D2, self.builder_E]
        permutation3 = [self.builder_A, self.builder_B2, self.builder_C, self.builder_D1, self.builder_E]
        permutation4 = [self.builder_A, self.builder_B2, self.builder_C, self.builder_D2, self.builder_E]

        self.permutations = [permutation1, permutation2, permutation3, permutation4]

        self.instructions_repo = InstructionsRepository()
        self.call_count_log = CallCountLog()
        self.manager = SmartCacheManager(self.instructions_repo, self.call_count_log)
        self.loader = CachingInstructionsLoader(manager=self.manager)

    def test_load_builder_models(self):
        models = self.loader.load_builder_models(self.permutations)
        self.assertEqual(4, len(models))
        for permutation in models:
            self.assertEqual(5, len(permutation))
            for model in permutation:
                self.assertIsInstance(model, BuilderInstructionModel)

    def test_load_builder_signatures(self):
        models = self.loader.load_builder_models(self.permutations)
        self.loader.load_prev_builder_descriptions(models)
        models: List[List[BuilderInstructionModel]]
        signatures = [[model.prev_builders for model in permutation] for permutation in models]
        self.assertEqual([['A', 'AB1', 'AB1C', 'AB1CD1', 'AB1CD1E'],
                          ['A', 'AB1', 'AB1C', 'AB1CD2', 'AB1CD2E'],
                          ['A', 'AB2', 'AB2C', 'AB2CD1', 'AB2CD1E'],
                          ['A', 'AB2', 'AB2C', 'AB2CD2', 'AB2CD2E']],
                         signatures)

    def test_map_instructions_to_signatures(self):
        models = self.loader.load_builder_models(self.permutations)
        self.loader.load_prev_builder_descriptions(models)
        self.loader.map_instructions_to_models(models)

        expected_instructions_first_iteration = {
            'A': 'function_cache',
            'AB1': 'function_cache',
            'AB2': 'function_cache',
            'AB1C': 'function_cache',
            'AB2C': 'function_cache',
            'AB1CD1': 'no cache',
            'AB1CD2': 'no cache',
            'AB2CD1': 'no cache',
            'AB2CD2': 'no cache',
            'AB1CD1E': 'no cache',
            'AB1CD2E': 'no cache',
            'AB2CD1E': 'no cache',
            'AB2CD2E': 'no cache',
        }

        expected_instructions_second_iteration = {
            'A': 'function_cache',
            'AB1': 'function_cache',
            'AB2': 'function_cache',
            'AB1C': 'function_cache',
            'AB2C': 'function_cache',
            'AB1CD1': 'no cache',
            'AB1CD2': 'no cache',
            'AB2CD1': 'no cache',
            'AB2CD2': 'no cache',
            'AB1CD1E': 'no cache',
            'AB1CD2E': 'no cache',
            'AB2CD1E': 'no cache',
            'AB2CD2E': 'no cache',
        }

        for permutation in models:
            for model in permutation:
                print(model.builder)
                prev_builders = model.prev_builders
                first_iteration_instruction = model.builder_instructions.get_instruction(0).instruction
                print(first_iteration_instruction)
                self.assertEqual(expected_instructions_first_iteration[prev_builders], first_iteration_instruction,
                                 f'Wrong instruction on first iteration for model: {str(model.builder.__hash__())}')
                second_iteration_instruction = model.builder_instructions.get_instruction(1).instruction
                print(second_iteration_instruction)
                self.assertEqual(expected_instructions_second_iteration[prev_builders], second_iteration_instruction,
                                 f'Wrong instruction on second iteration for model: {str(model.builder.__hash__())}')


class TestCachingInstructionsLoaderSimple(unittest.TestCase):

    def setUp(self) -> None:
        self.builder_A = mock.Mock(name='A', spec=Builder)
        self.builder_A.dependent_on = ['B']
        self.builder_A.builder_type = 'A'
        self.builder_A.__hash__ = mock.Mock()
        self.builder_A.__hash__.return_value = 'A'
        self.builder_A.build = mock.Mock()

        self.builder_B = mock.Mock(name='B', spec=Builder)
        self.builder_B.dependent_on = ['C']
        self.builder_B.builder_type = 'B'
        self.builder_B.__hash__ = mock.Mock()
        self.builder_B.__hash__.return_value = 'B'
        self.builder_B.build = mock.Mock()

        self.builder_C1 = mock.Mock(name='C1', spec=Builder)
        self.builder_C1.dependent_on = []
        self.builder_C1.builder_type = 'C'
        self.builder_C1.__hash__ = mock.Mock()
        self.builder_C1.__hash__.return_value = 'C1'
        self.builder_C1.build = mock.Mock()

        self.builder_C2 = mock.Mock(name='C2', spec=Builder)
        self.builder_C2.dependent_on = []
        self.builder_C2.builder_type = 'C'
        self.builder_C2.__hash__ = mock.Mock()
        self.builder_C2.__hash__.return_value = 'C2'
        self.builder_C2.build = mock.Mock()

        permutation1 = [self.builder_A, self.builder_B, self.builder_C1]
        permutation2 = [self.builder_A, self.builder_B, self.builder_C2]

        self.permutations = [permutation1, permutation2]

        self.instructions_repo = InstructionsRepository()
        self.call_count_log = CallCountLog()
        self.manager = SmartCacheManager(self.instructions_repo, self.call_count_log)
        self.loader = CachingInstructionsLoader(manager=self.manager)

    def test_load_builder_models(self):
        models = self.loader.load_builder_models(self.permutations)
        self.assertEqual(2, len(models))
        for permutation in models:
            self.assertEqual(3, len(permutation))
            for model in permutation:
                self.assertIsInstance(model, BuilderInstructionModel)

    def test_load_builder_signatures(self):
        models = self.loader.load_builder_models(self.permutations)
        self.loader.load_prev_builder_descriptions(models)
        models: List[List[BuilderInstructionModel]]
        signatures = [[model.prev_builders for model in permutation] for permutation in models]
        self.assertEqual([['A', 'AB', 'ABC1'], ['A', 'AB', 'ABC2']], signatures)

    def test_map_instructions_to_signatures(self):
        models = self.loader.load_builder_models(self.permutations)
        self.loader.load_prev_builder_descriptions(models)
        self.loader.map_instructions_to_models(models)

        expected_instructions_first_iteration = {
            'A': 'function_cache',
            'AB': 'function_cache',
            'ABC1': 'no cache',
            'ABC2': 'no cache',
        }

        expected_instructions_second_iteration = {
            'A': 'function_cache',
            'AB': 'function_cache',
            'ABC1': 'no cache',
            'ABC2': 'no cache',
        }

        for permutation in models:
            for model in permutation:
                prev_builders = model.prev_builders
                first_iteration_instruction = model.builder_instructions.get_instruction(0).instruction
                print(first_iteration_instruction)
                self.assertEqual(expected_instructions_first_iteration[prev_builders], first_iteration_instruction,
                                 f'Wrong instruction on first iteration for model: {str(model.builder.__hash__())}')
                second_iteration_instruction = model.builder_instructions.get_instruction(1).instruction
                print(second_iteration_instruction)
                self.assertEqual(expected_instructions_second_iteration[prev_builders], second_iteration_instruction,
                                 f'Wrong instruction on second iteration for model: {str(model.builder.__hash__())}')

if __name__ == '__main__':
    unittest.main()
