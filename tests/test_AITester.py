from AIBuilder.AITester import AITester
from unittest import TestCase, mock
import unittest


class AITesterTest(TestCase):

    def setUp(self):
        self.ai = mock.Mock('test_AITester.AbstractAI')
        self.ai.description = {'builder_1': {'ingredient_1': 1}}
        self.ai.train = mock.Mock(name='train')
        self.ai.get_name = mock.Mock(name='get_name')
        self.ai.get_name.return_value = 'name'
        self.ai.get_project_name = mock.Mock(name='get_project_name')
        self.ai.get_project_name.return_value = 'project'
        self.ai.get_log_dir = mock.Mock(name='get_log_dir')
        self.ai.get_log_dir.return_value = 'path'

        summizer = mock.patch('AITester.Summizer')
        summizer.log = mock.Mock(name='log')

        self.ai_tester = AITester(summizer=summizer)
        self.ai_tester.AI = self.ai

    def test_training(self):
        self.ai_tester.train_AI()
        self.ai.train.assert_called()
        self.assertEqual(2, self.ai_tester.summizer.log.call_count)

    def test_evaluation(self):
        self.ai_tester.log_testing_report = mock.Mock()
        self.ai_tester.summizer.summize = mock.Mock()
        self.ai_tester.determine_test_time()
        self.ai.evaluate = mock.Mock()
        self.ai.evaluate.return_value = {'result': 'test_dir'}

        self.ai_tester.evaluate_AI()

        self.assertEqual(2, self.ai_tester.summizer.log.call_count)
        self.ai_tester.create_model_dir_path = mock.Mock()
        self.ai.evaluate.assert_called_once()
        self.ai_tester.log_testing_report.assert_called_once()
        self.ai_tester.summizer.summize.assert_called_once()

    @mock.patch('AIBuilder.AITester.print')
    def test_print_results(self, print: mock.Mock):
        self.ai_tester.results = {'a': 'a'}
        self.ai_tester.print_evaluation_results()
        print.assert_called_once_with('a' + ': ' + str('a'))

    def test_log_testing_report(self):
        open = mock.mock_open()
        self.ai_tester.results = {'a': 'a'}
        self.ai_tester.test_time = 'testTime'
        file = mock.Mock()
        file.write = mock.Mock()
        open.return_value = file
        with mock.patch('AIBuilder.AITester.open', open, create=True):
            self.ai_tester.log_testing_report()
            open.assert_called_once_with('path/project/ai_reports.txt', mode='a')
            print_name = mock.call('\n--- AI: ' + 'name' + ' ---')
            print_time = mock.call('\n--- time: ' + 'testTime' + ' ---')
            print_report = mock.call('\n' + 'a' + ': ' + str('a'))
            print_newline = mock.call('\n')
            print_builder = mock.call('\nbuilder_1')
            print_spec = mock.call('\n - ingredient_1: 1')
            file.write.assert_has_calls(
                [print_name, print_time, print_report, print_newline, print_builder, print_spec],
                any_order=True)

    def test_is_unique(self):
        open = mock.mock_open()
        file = ['line1', 'description: 123', 'line3']
        open.return_value = file

        with mock.patch('AIBuilder.AITester.open', open, create=True):
            result = self.ai_tester.is_unique()
            self.assertTrue(result)

    def test_is_not_unique(self):
        open = mock.mock_open()
        file = ['line1', 'description: 4015285680072685342', 'line3']
        open.return_value = file

        with mock.patch('AIBuilder.AITester.open', open, create=True):
            result = self.ai_tester.is_unique()
            self.assertTrue(result)

    def test_stable_hash_description(self):
        result_hash = '986514f2494f256f444d9652abf742fc'
        description = {'a': 'a'}
        hash = AITester.stable_hash_description(description)
        self.assertEqual(result_hash, hash)


if __name__ == '__main__':
    unittest.main()