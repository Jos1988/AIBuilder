from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.Summizer import Summizer
from datetime import datetime
import hashlib


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

    def run_AI_test(self, ai: AbstractAI):
        self.set_AI(ai)
        if self.is_unique():
            self.train_AI()
            self.evaluate_AI()
        else:
            print('AI evaluated!')
            print(ai.description)

    def set_AI(self, ai: AbstractAI):
        self.AI = ai

    def is_unique(self) -> bool:
        report = self.open_report_file('r')
        description_hash = AITester.stable_hash_description(self.AI.description)
        for line in report:
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
        self.validate_test_time()
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
