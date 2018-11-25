from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.Summizer import Summizer, TimeSummizer
from unittest import TestCase, mock
import unittest
from datetime import datetime
import hashlib


# AbstractAITester
class AbstractAITester(ABC):

    @abstractmethod
    def train_AI(self):
        pass

    @abstractmethod
    def evaluate_AI(self):
        pass


# todo refactor write and print logic to writer and printer.


class AITester(AbstractAITester):
    AI: AbstractAI
    description_hash_label = '\n--- description hash: '

    def __init__(self, summizer: Summizer):
        self.summizer = summizer
        self.test_time = None
        self.results = {}

    def set_AI(self, ai: AbstractAI):
        self.AI = ai

    def is_unique(self) -> bool:
        report = self.open_report_file('r')
        description_hash = AITester.stable_hash_description(self.AI.description)
        # description_hash = hash(repr(self.AI.description))
        # print(description_hash)
        # description_hash = hash(repr(self.AI.description))
        # print(description_hash)
        for line in report:
            # print(line)
            # print(line.find(str(description_hash)))
            if -1 is not line.find(str(description_hash)):
                return False

        return True

    def train_AI(self):
        self.summizer.log('start training', None)
        self.determine_test_time()
        self.AI.train()
        self.summizer.log('finished training', None)

    def evaluate_AI(self):
        self.summizer.log('start evaluation', None)

        self.results = self.AI.evaluate()

        print()
        self.print_description()
        print()
        self.print_evaluation_results()
        self.log_testing_report()

        self.summizer.log('finished evaluation', None)
        self.summizer.summize()

    def print_evaluation_results(self):
        self.validate_results_set()
        for label, value in self.results.items():
            print(label + ': ' + str(value))

    def print_description(self):
        print('--- AI: ' + self.AI.get_name() + ' ---')
        print('--- time: ' + self.test_time + ' ---')
        for builder_name, description in self.AI.description.items():
            print(builder_name)

            if type(description) is not dict:
                print(' - ' + description)
                continue

            for element, value in description.items():
                print(' - ' + element + ' : ' + str(value))

    def log_testing_report(self):
        self.validate_results_set()
        self.validate_test_time()

        report = self.open_report_file('a')
        report.write('\n')
        report.write('\n--- AI: ' + self.AI.get_name() + ' ---')
        report.write('\n--- time: ' + self.test_time + ' ---')
        self.write_description(report)
        report.write('\n')
        self.write_results(report)

        report.close()

    def write_results(self, report):
        for label, value in self.results.items():
            report.write('\n' + label + ': ' + str(value))

    def write_description(self, report):
        description_hash = AITester.stable_hash_description(self.AI.description)
        report.write('\n--- description hash: ' + str(description_hash))
        for builder_name, description in self.AI.description.items():
            report.write('\n' + builder_name)

            if type(description) is not dict:
                report.write('\n - ' + description)
                continue

            for element, value in description.items():
                report.write('\n - ' + element + ': ' + str(value))

    def validate_results_set(self):
        assert type(self.results) is dict, 'Test results not set in AI tester.'

    def determine_test_time(self):
        self.test_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    def validate_test_time(self):
        assert type(self.test_time) is str, 'Test time not set in AITester.'

    def open_report_file(self, mode: str):
        path = self.AI.get_log_dir() + '/' + self.AI.get_project_name() + '/ai_reports.txt'
        report = open(path, mode=mode)
        return report

    @staticmethod
    def stable_hash_description(description: dict):
        description = repr(description)
        description = description.encode('utf-8')
        hash_result = hashlib.md5(description)

        return hash_result.hexdigest()


class AITesterTest(TestCase):

    def setUp(self):
        self.ai = mock.Mock('AITester.AbstractAI')
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
        self.ai.evaluate = mock.Mock()
        self.ai.evaluate.return_value = {'result': 'test_dir'}

        self.ai_tester.evaluate_AI()

        self.assertEqual(2, self.ai_tester.summizer.log.call_count)
        self.ai_tester.create_model_dir_path = mock.Mock()
        self.ai.evaluate.assert_called_once()
        self.ai_tester.log_testing_report.assert_called_once()
        self.ai_tester.summizer.summize.assert_called_once()

    @mock.patch('AITester.print')
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

        with mock.patch('AITester.open', open, create=True):
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

        with mock.patch('AITester.open', open, create=True):
            result = self.ai_tester.is_unique()
            self.assertTrue(result)

    def test_is_not_unique(self):
        open = mock.mock_open()
        file = ['line1', 'description: 4015285680072685342', 'line3']
        open.return_value = file

        with mock.patch('AITester.open', open, create=True):
            result = self.ai_tester.is_unique()
            self.assertTrue(result)

    def test_stable_hash_description(self):
        result_hash = '986514f2494f256f444d9652abf742fc'
        description = {'a': 'a'}
        hash = AITester.stable_hash_description(description)
        self.assertEqual(result_hash, hash)

        # report = self.open_report_file('r')
        # description_hash = hash(repr(self.ai.description))
        # print(report.read())
        # for line in report:
        #     if -1 is not line.find(str(description_hash)):
        #         return False
        #
        # return True



# todo: identify ai to check if it has already been trained and evaluated.

# class HardTestAITester(TestCase):
#
#     def test_log(self):
#         time_summizer = TimeSummizer()
#         tester = AITester('test_name', '../test_dir', summizer=time_summizer)
#
#         tester.determine_test_time()
#         print(tester.test_time)
#         tester.results = {'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd'}
#         tester.log_testing_report()


if __name__ == '__main__':
    unittest.main()
