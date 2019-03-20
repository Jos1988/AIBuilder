from AIBuilder.AIFactory.AIFactory import AIFactory
from AIBuilder.AIFactory.Builders import Builder
from AIBuilder.AITester import AITester
from AIBuilder.Dispatcher import Dispatcher
from AIBuilder.Summizer import TimeSummizer


class Kernel:

    def __init__(self, project_name: str, log_dir: str):
        self.project_name = project_name
        self.log_dir = log_dir
        self.factory = None
        self.tester = None
        self.dispatcher = None

    def boot(self):
        self.dispatcher = Dispatcher()
        self.dispatcher.dispatch(Dispatcher.PRE_BOOT)
        self.factory = AIFactory(
            builders=Builder.get_all_registered(),
            project_name=self.project_name,
            log_dir=self.log_dir)
        summizer = TimeSummizer()

        self.tester = AITester(summizer=summizer, evaluators=[])
        self.dispatcher.dispatch(Dispatcher.POST_BOOT)

    def run(self):
        # todo refactor to use the observers, remove logging from tester.
        self.dispatcher.dispatch(Dispatcher.PRE_RUN)
        while self.factory.has_next_ai():
            self.dispatcher.dispatch(Dispatcher.PRE_CREATE)
            self.factory.print_remaining_ai()
            model = self.factory.create_next_ai()
            self.tester.loadModel(model)
            self.dispatcher.dispatch(Dispatcher.POST_CREATE)

            if not self.tester.is_unique():
                self.tester.logModelNotUnique()

            self.dispatcher.dispatch(Dispatcher.PRE_TRAIN)
            self.tester.train_AI()
            self.dispatcher.dispatch(Dispatcher.POST_TRAIN)

            self.dispatcher.dispatch(Dispatcher.PRE_EVALUATE)
            self.tester.evaluate_AI()
            self.dispatcher.dispatch(Dispatcher.POST_EVALUATE)

        self.dispatcher.dispatch(Dispatcher.POST_RUN)
