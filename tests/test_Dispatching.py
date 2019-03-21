from unittest import TestCase, mock

from AIBuilder.Dispatching import KernelDispatcher, ModelNotUniqueObserver, PreCreateObserver, PreTrainObserver, \
    PostTrainObserver, PreEvaluationObserver, PostEvaluationObserver, Event


class KernelFullDispatchingTest(TestCase):

    def setUp(self):
        self.dispatcher = KernelDispatcher()
        self.ModelNotUniqueObserver = ModelNotUniqueObserver()
        self.ModelNotUniqueObserver.execute = mock.Mock()
        self.PreCreateObserver = PreCreateObserver()
        self.PreCreateObserver.execute = mock.Mock()
        self.PreTrainObserver = PreTrainObserver()
        self.PreTrainObserver.execute = mock.Mock()
        self.PostTrainObserver = PostTrainObserver()
        self.PostTrainObserver.execute = mock.Mock()
        self.PreEvaluationObserver = PreEvaluationObserver()
        self.PreEvaluationObserver.execute = mock.Mock()
        self.PostEvaluationObserver = PostEvaluationObserver()
        self.PostEvaluationObserver.execute = mock.Mock()

        self.dispatcher.addObserver(self.ModelNotUniqueObserver)
        self.dispatcher.addObserver(self.PreCreateObserver)
        self.dispatcher.addObserver(self.PreTrainObserver)
        self.dispatcher.addObserver(self.PostTrainObserver)
        self.dispatcher.addObserver(self.PreEvaluationObserver)
        self.dispatcher.addObserver(self.PostEvaluationObserver)

    def testDispatch(self):
        for event_name in KernelDispatcher.ALL_EVENTS:
            event = mock.Mock('Event')
            event.get_name = mock.Mock()
            event.name = event_name

            self.dispatcher.dispatch(event)

        self.ModelNotUniqueObserver.execute.assert_called_once()
        self.PreCreateObserver.execute.assert_called_once()
        self.PreTrainObserver.execute.assert_called_once()
        self.PostTrainObserver.execute.assert_called_once()
        self.PreEvaluationObserver.execute.assert_called_once()
        self.PostEvaluationObserver.execute.assert_called_once()
