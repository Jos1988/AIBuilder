from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI
from AIBuilder.Summizer import Summizer
from datetime import datetime
import hashlib
from _io import TextIOWrapper


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
        console_strategy = ConsolePrintStrategy()
        self.console_printer = Printer(console_strategy)
        self.report_printer = self.set_report_printer()
        self.description_hash = None

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

        return Printer(report_print_strategy)

    def set_AI(self, ai: AbstractAI):
        self.AI = ai

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

        self.summizer.log('finished evaluation', None)
        self.summizer.summize()

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


class PrintStrategy(ABC):

    @abstractmethod
    def print(self, text: str):
        pass

    @abstractmethod
    def new_line(self):
        pass

    def print_new_line(self, text: str):
        self.new_line()
        self.print(text)


class ConsolePrintStrategy(PrintStrategy):

    def print(self, text: str):
        print(text)

    def new_line(self):
        print()


class ReportPrintStrategy(PrintStrategy):

    def __init__(self, report: TextIOWrapper):
        self.report = report

    def print(self, text: str):
        self.report.write(text)

    def new_line(self):
        self.report.write('\n')

    def close_report(self):
        self.report.close()


class Printer:

    def __init__(self, strategy: PrintStrategy):
        self.output = strategy

    def separate(self,):
        self.line('==================================================================================================')

    def line(self, text):
        self.output.print_new_line(text)

    def print_ai_description(self, ai: AbstractAI, time_stamp: str = None, ai_hash: str = None):
        self.line('--- AI: ' + ai.get_name() + ' ---')
        if time_stamp is not None:
            self.line('--- time: ' + time_stamp + ' ---')

        if ai_hash is not None:
            self.line('--- description hash: ' + str(ai_hash))

        for builder_name, description in ai.description.items():
            self.line(builder_name)

            if type(description) is not dict:
                self.line(' - ' + description)
                continue

            for element, value in description.items():
                self.line(' - ' + element + ' : ' + str(value))

    def print_results(self, results: dict):
        for label, value in results:
            self.line(label + ': ' + str(value))
