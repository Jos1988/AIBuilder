from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from AIBuilder import AITester
from AIBuilder.AIFactory import AIFactory
from AIBuilder.AIFactory.Logging import CSVReader, MetaLogger, SummaryLogger
from AIBuilder.AIFactory.Printing import TesterPrinter, ConsolePrintStrategy, ReportPrintStrategy, \
    FactoryPrinter

class Session:

    def __init__(self, project_name: str, log_dir: str):
        self.session_data = {}
        self.log_dir = log_dir
        self.project_name = project_name

    def set_meta_logging(self, model_attributes: List[str], metrics: List[str], discriminator: str):
        data = {'attributes': model_attributes, 'metrics': metrics, 'discriminator': discriminator}
        self.session_data['meta_logging'] = data


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

    def __init__(self, event_name: str, session: Session, tester: AITester):
        self.session = session
        self.event_name = event_name
        self.tester = tester

    @property
    def name(self) -> str:
        return self.event_name


class FactoryEvent(Event):

    def __init__(self, event_name: str, session: Session, factory: AIFactory):
        self.session = session
        self.event_name = event_name
        self.factory = factory

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
        printer.print_ai_description(ai=event.tester.ml_model, time_stamp=event.tester.test_time,
                                     ai_hash=event.tester.description_hash)
        printer.separate()


class PreRunObserver(Observer):
    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.PRE_RUN]

    def execute(self, event: Event):
        pass


class PreRunLogObserver(Observer):
    meta_log_file_name = 'meta_log.csv'
    summary_file_name = 'summary.csv'

    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.PRE_RUN]

    def check_logfile(self, log_file_path: Path, event: TesterEvent):
        if not log_file_path.is_file():
            return

        self.check_logfile_compatible(log_file_path=log_file_path, event=event)

    @abstractmethod
    def execute(self, event: Event):
        pass

    def check_logfile_compatible(self, log_file_path: Path, event: TesterEvent):
        attributes = event.session.session_data['meta_logging']['attributes']
        metrics = event.session.session_data['meta_logging']['metrics']
        reader = CSVReader(
            attribute_names=attributes,
            metric_names=metrics,
            discriminator=event.session.session_data['meta_logging']['discriminator']
        )

        if False is reader.check_compatible(log_file_path):
            raise AssertionError('Meta log file not compatible, expecting following headers: {}'.format(reader.all_names))

    def get_path(self, event: Event, file_name: str) -> str:
        return event.session.log_dir + '/' + event.session.project_name + '/' + file_name

    def load_meta_logger(self, event, path: Path):
        meta_logger = MetaLogger(
            log_attributes=event.session.session_data['meta_logging']['attributes'],
            log_metrics=event.session.session_data['meta_logging']['metrics'],
            discrimination_value=event.session.session_data['meta_logging']['discriminator'],
            log_file_path=path
        )

        return meta_logger


class PreRunLoadMetaLogger(PreRunLogObserver):

    def execute(self, event: Event):
        assert type(event) is TesterEvent
        event: TesterEvent

        meta_log_file_path = Path(self.get_path(event, self.meta_log_file_name))

        if meta_log_file_path.is_file():
            self.check_logfile_compatible(meta_log_file_path, event=event)

        meta_logger = self.load_meta_logger(event, path=meta_log_file_path)
        event.session.meta_logger = meta_logger

    def load_meta_logger(self, event, path: Path):
        meta_logger = MetaLogger(
            log_attributes=event.session.session_data['meta_logging']['attributes'],
            log_metrics=event.session.session_data['meta_logging']['metrics'],
            discrimination_value=event.session.session_data['meta_logging']['discriminator'],
            log_file_path=path
        )

        return meta_logger


class PreRunLoadSummaryLogger(PreRunLogObserver):

    def execute(self, event: Event):
        assert type(event) is TesterEvent
        event: TesterEvent

        meta_log_file_path = Path(self.get_path(event, self.meta_log_file_name))
        self.check_logfile(meta_log_file_path, event)
        summary_path = Path(self.get_path(event, self.summary_file_name))
        self.check_logfile(summary_path, event)

        meta_logger = self.load_summary_logger(event, meta_log_path=meta_log_file_path, summary_path=summary_path)
        event.session.summary_logger = meta_logger

    def load_summary_logger(self, event, meta_log_path: Path, summary_path: Path):
        meta_logger = SummaryLogger(
            log_attributes=event.session.session_data['meta_logging']['attributes'],
            log_metrics=event.session.session_data['meta_logging']['metrics'],
            discrimination_value=event.session.session_data['meta_logging']['discriminator'],
            log_file_path=meta_log_path,
            summary_log_file_path=summary_path
        )

        return meta_logger


class PostRunObserver(Observer):

    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.POST_RUN]

    def execute(self, event: Event):
        assert type(event) is TesterEvent
        event: TesterEvent
        pass


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

        self.display_console_update(event)

    def display_console_update(self, event):
        printer = TesterPrinter(ConsolePrintStrategy())
        printer.separate()
        printer.line('Finished Evaluation')
        printer.separate()
        event.tester.validate_test_time()
        printer.print_ai_description(ai=event.tester.ml_model, time_stamp=event.tester.test_time,
                                     ai_hash=event.tester.description_hash)
        event.tester.validate_results_set()
        printer.print_results(event.tester.ml_model.results)
        event.tester.summizeTime(ConsolePrintStrategy())


class PostEvaluationLogObserver(Observer):
    @property
    def observed_event_names(self) -> List[str]:
        return [KernelDispatcher.POST_EVALUATE]

    def execute(self, event: Event):
        assert type(event) is TesterEvent
        event: TesterEvent

        self.update_report(event)
        self.update_meta_report(event)
        self.update_summary(event)
        event.session.meta_logger.save_logged_data()
        event.session.summary_logger.save_logged_data()

    def update_report(self, event):
        report = event.tester.get_report_file('a')
        report_printer = TesterPrinter(ReportPrintStrategy(report=report))
        report_printer.line('')
        report_printer.print_ai_description(
            ai=event.tester.ml_model, time_stamp=event.tester.test_time, ai_hash=event.tester.description_hash)
        report_printer.line('')
        report_printer.print_results(event.tester.ml_model.results)
        report_printer.separate()
        event.tester.summizeTime(report_printer.output)
        report_printer.separate()
        event.tester.summizer.reset()
        report_printer.output.close_report()

    def update_meta_report(self, event):
        event.session.meta_logger.add_ml_model(event.tester.ml_model)

    def update_summary(self, event):
        event.session.summary_logger.add_ml_model(event.tester.ml_model)
