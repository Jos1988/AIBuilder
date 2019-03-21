from AIBuilder.AIFactory.AIFactory import AIFactory
from AIBuilder.AIFactory.Builders import Builder
from AIBuilder.AITester import AITester
from AIBuilder.Dispatching import KernelDispatcher, FactoryEvent, TesterEvent, ModelNotUniqueObserver, \
    PreCreateObserver, \
    PreTrainObserver, PostTrainObserver, PreEvaluationObserver, PostEvaluationObserver
from AIBuilder.Summizer import TimeSummizer


class Kernel:

    def __init__(self, project_name: str, log_dir: str):
        self.project_name = project_name
        self.log_dir = log_dir
        self.factory = None
        self.tester = None
        self.dispatcher = None

    def boot(self):
        self.dispatcher = KernelDispatcher()
        self.dispatcher.addObserver(ModelNotUniqueObserver())
        self.dispatcher.addObserver(PreCreateObserver())
        self.dispatcher.addObserver(PreTrainObserver())
        self.dispatcher.addObserver(PostTrainObserver())
        self.dispatcher.addObserver(PreEvaluationObserver())
        self.dispatcher.addObserver(PostEvaluationObserver())
        # self.dispatcher.dispatch(KernelDispatcher.PRE_BOOT) pre-boot hook
        self.factory = AIFactory(
            builders=Builder.get_all_registered(),
            project_name=self.project_name,
            log_dir=self.log_dir)
        summizer = TimeSummizer()

        self.tester = AITester(summizer=summizer, evaluators=[])
        # self.dispatcher.dispatch(KernelDispatcher.POST_BOOT) post-boot hook

    def run(self):
        # self.dispatcher.dispatch(KernelDispatcher.PRE_RUN) pre-run hook.
        while self.factory.has_next_ai():
            self.doCreateModel()

            if not self.tester.is_unique():
                self.ModelNotUnique()
                continue

            self.doTrainModel()

            self.doEvaluateModel()

        # self.dispatcher.dispatch(KernelDispatcher.POST_RUN) post-run hook.

    def doEvaluateModel(self):
        pre_evaluate_event = TesterEvent(event_name=KernelDispatcher.PRE_EVALUATE, tester=self.tester)
        self.dispatcher.dispatch(pre_evaluate_event)

        self.tester.evaluate_AI()

        post_evaluate_event = TesterEvent(event_name=KernelDispatcher.POST_EVALUATE, tester=self.tester)
        self.dispatcher.dispatch(post_evaluate_event)

    def doTrainModel(self):
        pre_train_event = TesterEvent(event_name=KernelDispatcher.PRE_TRAIN, tester=self.tester)
        self.dispatcher.dispatch(pre_train_event)

        self.tester.train_AI()

        post_train_event = TesterEvent(event_name=KernelDispatcher.POST_TRAIN, tester=self.tester)
        self.dispatcher.dispatch(post_train_event)

    def ModelNotUnique(self):
        not_unique_event = TesterEvent(event_name=KernelDispatcher.MODEL_NOT_UNIQUE, tester=self.tester)
        self.dispatcher.dispatch(not_unique_event)

    def doCreateModel(self):
        pre_create_event = FactoryEvent(event_name=KernelDispatcher.PRE_CREATE, factory=self.factory)
        self.dispatcher.dispatch(pre_create_event)

        model = self.factory.create_next_ai()
        self.tester.loadModel(model)

        # self.dispatcher.dispatch(KernelDispatcher.POST_CREATE) post-create hook.
