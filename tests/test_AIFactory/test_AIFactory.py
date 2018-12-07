from AIBuilder.AIFactory.Builders import Builder
from AIBuilder.Data import MetaData
import unittest
from unittest import mock, TestCase
from AIBuilder.AIFactory import AIFactory


class TestAIFactory(TestCase):

    def setUp(self):
        self.factory = AIFactory.AIFactory(project_name='test', log_dir='test/dir')

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
        self.assertEqual(self.factory.builders_sorted, [self.builder_C, self.builder_B, self.builder_A])

        self.builder_A.build.assert_called_once_with(self.artie)
        self.builder_B.build.assert_called_once_with(self.artie)
        self.builder_C.build.assert_called_once_with(self.artie)

    def test_add_alternative_builder(self):
        pass

    def test_cycle_permutations(self):
        pass

    def test_generate_permutations(self):
        pass


if __name__ == '__main__':
    unittest.main()
