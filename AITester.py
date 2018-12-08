from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.AIFactory.Printing import ConsolePrintStrategy, TesterPrinter, ReportPrintStrategy
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


class AITester(AbstractAITester):
    AI: AbstractAI

    def __init__(self, summizer: Summizer):
        self.summizer = summizer
        self.test_time = None
        self.results = {}
        self.console_print_strategy = ConsolePrintStrategy()
        self.console_printer = TesterPrinter(self.console_print_strategy)
        self.description_hash = None
        self.report_printer = None

    def run_AI_test(self, ai: AbstractAI):
        self.set_AI(ai)
        self.description_hash = self.stable_hash_description(ai.description)

        if self.is_unique():
            self.train_AI()
            self.evaluate_AI()
        else:
            self.console_printer.line('')
            self.console_printer.separate()
            self.console_printer.line('AI already evaluated')
            self.print_description()

            self.console_printer.separate()

    def set_report_printer(self):
        report = self.open_report_file('a')
        report_print_strategy = ReportPrintStrategy(report=report)

        self.report_printer = TesterPrinter(report_print_strategy)

    def set_AI(self, ai: AbstractAI):
        self.AI = ai
        self.set_report_printer()

    def is_unique(self) -> bool:
        report = self.open_report_file('r')
        for line in report:
            if -1 is not line.find(str(self.description_hash)):
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

        self.print_description()
        self.print_results()
        self.log_testing_report()
        self.report_printer.output.close_report()

        self.summizer.log('finished evaluation', None)
        self.summizer.summize(self.console_print_strategy)
        self.summizer.reset()

    def print_results(self):
        self.console_printer.separate()
        self.validate_results_set()
        self.console_printer.print_results(self.results)

    def print_description(self):
        self.console_printer.separate()
        self.console_printer.print_ai_description(
            ai=self.AI, time_stamp=self.test_time, ai_hash=self.description_hash)

    def log_testing_report(self):
        self.report_printer.line('')
        self.report_printer.print_ai_description(
            ai=self.AI, time_stamp=self.test_time, ai_hash=self.description_hash)
        self.report_printer.line('')
        self.report_printer.print_results(self.results)
        self.report_printer.separate()
        self.summizer.summize(self.report_printer.output)
        self.report_printer.separate()

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
