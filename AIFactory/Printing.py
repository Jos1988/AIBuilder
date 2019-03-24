from _io import TextIOWrapper
from abc import ABC, abstractmethod
from AIBuilder.AI import AbstractAI


class PrintStrategy(ABC):

    @abstractmethod
    def print(self, text: str):
        pass

    @abstractmethod
    def new_line(self):
        pass

    @abstractmethod
    def print_new_line(self, text: str):
        self.new_line()
        self.print(text)


class ConsolePrintStrategy(PrintStrategy):

    def print(self, text: str):
        print(text, end='')

    def new_line(self):
        print()

    def print_new_line(self, text: str):
        print(text)


class ReportPrintStrategy(PrintStrategy):

    def __init__(self, report: TextIOWrapper):
        self.report = report

    def print(self, text: str):
        self.report.write(text)

    def new_line(self):
        self.report.write('\n')

    def print_new_line(self, text: str):
        self.new_line()
        self.print(text)

    def close_report(self):
        self.report.close()


class Printer:
    def __init__(self, strategy: PrintStrategy):
        self.output = strategy

    def line(self, text):
        self.output.print_new_line(text)

    def separate(self):
        self.line('==================================================================================================')


class TesterPrinter(Printer):

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
                self.line(' - ' + element + ': ' + str(value))

    def print_results(self, results: dict):
        for label, value in results.items():
            self.line(label + ': ' + str(value))


class FactoryPrinter(Printer):

    def print_remaining_ai(self, remaining_ai: int):
        self.separate()
        self.line('--- {} AIs remaining. ---'.format(remaining_ai))
        self.separate()
