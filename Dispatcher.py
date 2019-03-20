from abc import ABC, abstractmethod
from typing import List


class Observer(ABC):

    @property
    @abstractmethod
    def observed_events(self) -> List[str]:
        pass

    def observes(self, event: str) -> bool:
        return event in self.observed_events

    @abstractmethod
    def execute(self):
        pass


class Dispatcher:
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

    ALL_EVENTS = [PRE_BOOT, POST_BOOT, PRE_RUN, POST_RUN, PRE_CREATE, POST_CREATE, PRE_TRAIN, POST_TRAIN, PRE_EVALUATE,
                  POST_EVALUATE]

    observers = []

    def dispatch(self, event: str):
        assert self.validateEvent(event=event), '{} is not registered as a valid event.'.format(event)
        for observer in self.observers:
            if observer.observes(event):
                observer.execute()

    def addObserver(self, observer: Observer):
        if not self.hasObserver(observer):
            self.observers.append(observer)

    def hasObserver(self, observer: Observer) -> bool:
        return observer in self.observers

    def removeObserver(self, observer: Observer):
        if self.hasObserver(observer):
            self.observers.remove(observer)

    def validateEvent(self, event: str) -> bool:
        return event in self.ALL_EVENTS
