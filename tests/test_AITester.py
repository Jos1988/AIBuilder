from AIBuilder.AITester import AITester
from unittest import TestCase, mock
import unittest


class AITesterTest(TestCase):

    def setUp(self):
        self.ai = mock.Mock('test_AITester.AbstractAI')
        self.ai.description = {'builder_1': {'ingredient_1': 1}}
        self.ai.train = mock.Mock(name='train')
        self.ai.name = mock.Mock(name='get_name')
        self.ai.name.return_value = 'name'
        self.ai.get_project_name = mock.Mock(name='get_project_name')
        self.ai.get_project_name.return_value = 'project'
        self.ai.get_log_dir = mock.Mock(name='get_log_dir')
        self.ai.get_log_dir.return_value = 'path'

        summizer = mock.patch('AITester.Summizer')
        summizer.log = mock.Mock(name='log')
        summizer.summize = mock.Mock(name='summize')

        self.ai_tester = AITester(summizer=summizer)
        self.ai_tester.ml_model = self.ai

    def test_training(self):
        self.ai_tester.train_AI()
        self.ai.train.assert_called()
        self.assertEqual(2, self.ai_tester.summizer.log.call_count)

    def test_evaluation(self):
        self.ai_tester.determine_test_time()
        self.ai.evaluate = mock.Mock()
        self.ai.evaluate.return_value = {'result': 'test_dir'}

        self.ai_tester.evaluate_AI()

        self.assertEqual(2, self.ai_tester.summizer.log.call_count)
        self.ai_tester.create_model_dir_path = mock.Mock()
        self.ai.evaluate.assert_called_once()

    def test_stable_hash_description(self):
        result_hash = '986514f2494f256f444d9652abf742fc'
        description = {'a': 'a'}
        hash = AITester.stable_hash_description(description)
        self.assertEqual(result_hash, hash)


if __name__ == '__main__':
    unittest.main()
