from abc import ABC, abstractmethod
from typing import List

from AIBuilder import AITester
from AIBuilder.AIFactory import AIFactory
from AIBuilder.AIFactory.Printing import TesterPrinter, ConsolePrintStrategy, ReportPrintStrategy, \
    FactoryPrinter


class Event(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class Observer(ABC):

    @property
    @abstractmethod
    def observed_event_names(self) -> List[str]:
        pass

    def observes(self, event_name: str) -> bool:
        return event_name in self.observed_event_names

    @abstractmethod
    def execute(self, event: Event):
        pass


class Dispatcher(ABC):
    observers = []

    @abstractmethod
    def dispatch(self, event: Event):
        pass

    def addObserver(self, observer: Observer):
        if not self.hasObserver(observer):
            self.observers.append(observer)

    def hasObserver(self, observer: Observer) -> bool:
        return observer in self.observers

    def removeObserver(self, observer: Observer):
        if self.hasObserver(observer):
            self.observers.remove(observer)


class KernelDispatcher(Dispatcher):
    PRE_BOOT = 'PRE_BOOT'
    POST_BOOT = 'POST_BOOT'

    PRE_RUN = 'PRE_RUN'
    POST_RUN = 'POST_RUN'

    PRE_CREATE = 'PRE_CREATE'
    POST_CREATE = 'POST_CREATE'

    PRE_TRAIN = 'PRE_TRAIN'
    POST_TRAIN = 'POST_TRAIN'

    PRE_EVALUATE = 'PRE_EVALUATE'
    POST_EVALUATE = 'POST_EVALUATE'

    MODEL_NOT_UNIQUE = 'MODEL_NOT_UNIQUE'

    ALL_EVENTS = [PRE_BOOT, POST_BOOT, PRE_RUN, POST_RUN, PRE_CREATE, POST_CREATE, PRE_TRAIN, POST_TRAIN, PRE_EVALUATE,
                  POST_EVALUATE, MODEL_NOT_UNIQUE]

    def dispatch(self, event: Event):
        assert self.validateEvent(event=event.name), '{} is not registered as a valid event.'.format(event.name)
        for observer in self.observers:
            if observer.observes(event.name):
                observer.execute(event)

    def validateEvent(self, event: str) -> bool:
        return event in self.ALL_EVENTS


class TesterEvent(Event):
    tester: AITester

    def __init__(self, event_name: str, tester: AITester):
        self.event_name = event_name
        self.tester = tester

    @property
    def name(self) -> str:
        return self.event_name


class FactoryEvent(Event):

    def __init__(self, event_name, factory: AIFactory):
        self.factory = factory
        self.event_name = event_name

    @property
    def name(self) -> str:
        return self.event_name


class ModelNotUniqueObserver(Observer):
    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.MODEL_NOT_UNIQUE]

    def execute(self, event: Event):
        assert type(event) is TesterEvent
        event: TesterEvent
        printer = TesterPrinter(ConsolePrintStrategy())

        printer.line('')
        printer.separate()
        printer.line('AI already evaluated')

        printer.separate()
        printer.print_ai_description(ai=event.tester.AI, time_stamp=event.tester.test_time,
                                     ai_hash=event.tester.description_hash)
        printer.separate()


class PreCreateObserver(Observer):
    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.PRE_CREATE]

    def execute(self, event: Event):
        assert type(event) is FactoryEvent
        event: FactoryEvent
        printer = FactoryPrinter(ConsolePrintStrategy())

        printer.print_remaining_ai(event.factory.count_remaining_models())


class PreTrainObserver(Observer):
    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.PRE_TRAIN]

    def execute(self, event: Event):
        assert type(event) is TesterEvent
        event: TesterEvent
        printer = TesterPrinter(ConsolePrintStrategy())
        printer.separate()
        printer.line('Start Training')
        printer.separate()


class PostTrainObserver(Observer):
    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.POST_TRAIN]

    def execute(self, event: Event):
        assert type(event) is TesterEvent
        event: TesterEvent
        printer = TesterPrinter(ConsolePrintStrategy())
        printer.separate()
        printer.line('Finished Training')
        printer.separate()


class PreEvaluationObserver(Observer):
    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.PRE_EVALUATE]

    def execute(self, event: Event):
        assert type(event) is TesterEvent
        event: TesterEvent
        printer = TesterPrinter(ConsolePrintStrategy())
        printer.separate()
        printer.line('Start Evaluation')
        printer.separate()


class PostEvaluationObserver(Observer):
    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.POST_EVALUATE]

    def execute(self, event: Event):
        assert type(event) is TesterEvent
        event: TesterEvent

        printer = TesterPrinter(ConsolePrintStrategy())
        printer.separate()
        printer.line('Finished Evaluation')
        printer.separate()
        event.tester.validate_test_time()
        printer.print_ai_description(ai=event.tester.AI, time_stamp=event.tester.test_time,
                                     ai_hash=event.tester.description_hash)

        event.tester.validate_results_set()
        printer.print_results(event.tester.results)
        event.tester.summizeTime(ConsolePrintStrategy())

        report = event.tester.get_report_file('a')
        report_printer = TesterPrinter(ReportPrintStrategy(report=report))

        report_printer.line('')
        report_printer.print_ai_description(
            ai=event.tester.AI, time_stamp=event.tester.test_time, ai_hash=event.tester.description_hash)
        report_printer.line('')
        report_printer.print_results(event.tester.results)
        report_printer.separate()
        event.tester.summizeTime(report_printer.output)
        report_printer.separate()

        event.tester.summizer.reset()
        report_printer.output.close_report()
