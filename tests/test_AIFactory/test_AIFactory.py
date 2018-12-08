from AIBuilder.AIFactory.Builders import Builder
from AIBuilder.Data import MetaData
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


if __name__ == '__main__':
    unittest.main()
