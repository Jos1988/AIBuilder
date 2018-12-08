import time
from abc import ABC, abstractmethod
from AIBuilder.AIFactory.Printing import PrintStrategy


class Summizer(ABC):

    @abstractmethod
    def log(self, label: str, value):
        pass

    @abstractmethod
    def summize(self, printer: PrintStrategy):
        pass

    @abstractmethod
    def reset(self):
        pass


class TimeSummizer(Summizer):
    time_log = dict()
    start_time = None

    def __init__(self):
        self.start_time = time.time()

    def start_time_log(self):
        self.start_time = time.time()

    def log(self, label, value):
        """ log given time and if no time given log current time

        :param print:
        :param label:
        :param value:
        :return:
        """
        if self.start_time is None:
            self.start_time_log()

        if value is None:
            value = time.time()

        self.time_log[label] = value

    def summize(self, printer: PrintStrategy):
        """ print summary of logged times

        :return:
        """
        if self.start_time is None:
            print('!!! no time logging, start time not set.')
            return

        now = time.time()
        previous_time = self.start_time
        printer.new_line()
        for label, time_stamp in self.time_log.items():
            elapsed = time_stamp - previous_time
            printer.print_new_line(label + ': ' + str(elapsed))
            previous_time = time_stamp

        printer.print_new_line('--------------------')
        elapsed = now - self.start_time
        printer.print_new_line('total: ' + str(elapsed) + '\b')

    def reset(self):
        self.start_time = None
