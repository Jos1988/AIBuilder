from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.Summizer import Summizer
from unittest import TestCase, mock
from datetime import datetime


# AbstractAITester
class AbstractAITester(ABC):

    @abstractmethod
    def train_AI(self):
        pass

    @abstractmethod
    def evaluate_AI(self):
        pass


class AITester(AbstractAITester):
    AI: AbstractAI

    def __init__(self, test_name: str, log_dir: str, summizer: Summizer):
        self.summizer = summizer
        self.test_name = test_name
        self.test_time = None
        self.log_dir = log_dir
        self.results = {}

    def set_AI(self, ai: AbstractAI):
        self.AI = ai

    def train_AI(self):
        self.summizer.log('start training', None)
        self.AI.train()
        self.summizer.log('finished training', None)

    def evaluate_AI(self):
        self.summizer.log('start evaluation', None)

        model_dir = self.create_model_dir_path()
        self.results = self.AI.evaluate(model_dir=model_dir)

        self.print_evaluation_results()
        self.log_testing_report()

        self.summizer.log('finished evaluation', None)
        self.summizer.summize()

    def print_evaluation_results(self):
        self.validate_results_set()
        for label, value in self.results.items():
            print(label + ': ' + str(value))

    def log_testing_report(self):
        report_file = self.create_report_file_path()

        self.validate_results_set()
        report = open(report_file, 'a')
        report.write('\n--- ' + self.test_time + ' ---')
        for label, value in self.results.items():
            report.write('\n' + label + ': ' + str(value))

    def validate_results_set(self):
        assert type(self.results) is dict, 'Test results not set in AI tester.'

    def determine_test_time(self):
        self.test_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    def validate_test_time(self):
        assert self.test_time is str, 'Test time not set in tester, cannot render filename.'

    def create_model_dir_path(self):
        self.validate_test_time()

        return self.log_dir + '/' + self.test_name + '/tensor_board/' + self.test_time

    def create_report_file_path(self):
        self.validate_test_time()

        return self.log_dir + '/' + self.test_name + '/ai_reports.txt'


class AITesterTest(TestCase):

    def setUp(self):
        self.ai = mock.Mock('AITester.AbstractAI')
        self.ai.train = mock.Mock(name='train')

        summizer = mock.patch('AITester.Summizer')
        summizer.log = mock.Mock(name='log')

        dir_path = 'path/'
        self.ai_tester = AITester(log_dir=dir_path, test_name='test', summizer=summizer)
        self.ai_tester.AI = self.ai

    def test_training(self):
        self.ai_tester.train_AI()
        self.ai.train.assert_called()
        self.assertEqual(2, self.ai_tester.summizer.log.call_count)

    def test_evaluation(self):
        self.ai_tester.create_model_dir_path = mock.Mock()
        self.ai_tester.create_model_dir_path.return_value = 'test_dir/'
        self.ai_tester.log_testing_report = mock.Mock()
        self.ai_tester.summizer.summize = mock.Mock()
        self.ai.evaluate = mock.Mock()
        self.ai.evaluate.return_value = {'result': 'test'}

        self.ai_tester.evaluate_AI()

        self.assertEqual(2, self.ai_tester.summizer.log.call_count)
        self.ai_tester.create_model_dir_path = mock.Mock()
        self.ai.evaluate.assert_called_with(model_dir='test_dir/')
        self.ai_tester.log_testing_report.assert_called_once()
        self.ai_tester.summizer.summize.assert_called_once()

    @mock.patch('AITester.print')
    def test_print_results(self, print: mock.Mock):
        self.ai_tester.results = {'a': 'a'}
        self.ai_tester.print_evaluation_results()
        print.assert_called_once_with('a' + ': ' + str('a'))

    def test_log_testing_report(self):
        open = mock.mock_open()
        self.ai_tester.create_report_file_path = mock.Mock()

        self.ai_tester.create_report_file_path.return_value = 'path/to/file'
        self.ai_tester.results = {'a': 'a'}
        self.ai_tester.test_time = 'testTime'
        file = mock.Mock()
        file.write = mock.Mock()
        open.return_value = file

        with mock.patch('AITester.open', open, create=True):
            self.ai_tester.log_testing_report()
            open.assert_called_once_with('path/to/file', 'a')
            call1 = mock.call('\n--- ' + 'testTime' + ' ---')
            call2 = mock.call('\n' + 'a' + ': ' + str('a'))
            file.write.assert_has_calls([call1, call2])
