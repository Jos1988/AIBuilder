import time


class TimeSummizer:
    time_log = dict()
    start_time = None

    def __init__(self):
        self.start_time = time.time()

    def start_time_log(self):
        self.start_time = time.time()

    def log_time(self, name, log_time=None, print_now=False):
        """ log given time and if no time given log current time

        :param print:
        :param name:
        :param log_time:
        :return:
        """
        # todo  refactor to function and raise warning.
        if self.start_time is None:
            raise Exception('!!! no time logging, start time not set.')

        if log_time is None:
            log_time = time.time()

        self.time_log[name] = log_time

        if print_now is True:
            elapsed = log_time - self.start_time
            print('\b' + name + ':' + str(elapsed) + '\b')

    def summize_times(self, total=True):
        """ print summary of logged times

        :return:
        """
        # todo  refactor to function and raise warning.
        if self.start_time is None:
            print('!!! no time logging, start time not set.')
            return

        now = time.time()
        previous_time = self.start_time
        print('\b')
        for label, time_stamp in self.time_log.items():
            elapsed = time_stamp - previous_time
            print(label + ': ' + str(elapsed))
            previous_time = time_stamp

        if total is True:
            print('--------------------')
            elapsed = now - self.start_time
            print('total: ' + str(elapsed) + '\b')
