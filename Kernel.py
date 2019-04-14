from AIBuilder.AIFactory.AIFactory import AIFactory
from AIBuilder.AIFactory.Builders import Builder
from AIBuilder.AITester import AITester, GainBasedFeatureImportance, AccuracyBaselineDiff
from AIBuilder.Dispatching import KernelDispatcher, FactoryEvent, TesterEvent, ModelNotUniqueObserver, \
    PreCreateObserver, \
    PreTrainObserver, PostTrainObserver, PreEvaluationObserver, PostEvaluationObserver, Session, PreRunObserver, \
    PostRunObserver, PreRunLogObserver, PostEvaluationLogObserver
from AIBuilder.Summizer import TimeSummizer


class Kernel:

    def __init__(self, session: Session, **kwargs):
        self.session = session
        self.factory = None
        self.tester = None
        self.dispatcher = None
        self.no_log = False

        if 'no_log' in kwargs:
            self.no_log = kwargs['no_log']

    def boot(self):
        self.dispatcher = KernelDispatcher()
        self.dispatcher.addObserver(PreRunObserver())
        self.dispatcher.addObserver(ModelNotUniqueObserver())
        self.dispatcher.addObserver(PreCreateObserver())
        self.dispatcher.addObserver(PreTrainObserver())
        self.dispatcher.addObserver(PostTrainObserver())
        self.dispatcher.addObserver(PreEvaluationObserver())
        self.dispatcher.addObserver(PostEvaluationObserver())
        self.dispatcher.addObserver(PostRunObserver())

        if True is not self.no_log:
            self.dispatcher.addObserver(PreRunLogObserver())
            self.dispatcher.addObserver(PostEvaluationLogObserver())

        # self.dispatcher.dispatch(KernelDispatcher.PRE_BOOT) pre-boot hook
        self.factory = AIFactory(
            builders=Builder.get_all_registered(),
            project_name=self.session.project_name,
            log_dir=self.session.log_dir)
        summizer = TimeSummizer()

        feature_importance = GainBasedFeatureImportance()
        accuracy_diff = AccuracyBaselineDiff()

        self.tester = AITester(summizer=summizer, evaluators=[feature_importance, accuracy_diff])
        # self.dispatcher.dispatch(KernelDispatcher.POST_BOOT) post-boot hook

    def run(self):
        pre_run_event = TesterEvent(event_name=KernelDispatcher.PRE_RUN, session=self.session, tester=self.tester)
        self.dispatcher.dispatch(pre_run_event)
        while self.factory.has_next_ai():
            self.doCreateModel()

            if not self.tester.is_unique():
                self.ModelNotUnique()
                continue

            self.doTrainModel()

            self.doEvaluateModel()

        post_run_event = TesterEvent(event_name=KernelDispatcher.POST_RUN, session=self.session, tester=self.tester)
        self.dispatcher.dispatch(post_run_event)

    def doEvaluateModel(self):
        pre_evaluate_event = TesterEvent(event_name=KernelDispatcher.PRE_EVALUATE, session=self.session,
                                         tester=self.tester)
        self.dispatcher.dispatch(pre_evaluate_event)

        self.tester.evaluate_AI()

        post_evaluate_event = TesterEvent(event_name=KernelDispatcher.POST_EVALUATE, session=self.session,
                                          tester=self.tester)
        self.dispatcher.dispatch(post_evaluate_event)

    def doTrainModel(self):
        pre_train_event = TesterEvent(event_name=KernelDispatcher.PRE_TRAIN, session=self.session, tester=self.tester)
        self.dispatcher.dispatch(pre_train_event)

        self.tester.train_AI()

        post_train_event = TesterEvent(event_name=KernelDispatcher.POST_TRAIN, session=self.session, tester=self.tester)
        self.dispatcher.dispatch(post_train_event)

    def ModelNotUnique(self):
        not_unique_event = TesterEvent(event_name=KernelDispatcher.MODEL_NOT_UNIQUE, session=self.session,
                                       tester=self.tester)
        self.dispatcher.dispatch(not_unique_event)

    def doCreateModel(self):
        pre_create_event = FactoryEvent(event_name=KernelDispatcher.PRE_CREATE, session=self.session,
                                        factory=self.factory)
        self.dispatcher.dispatch(pre_create_event)

        model = self.factory.create_next_ai()
        self.tester.loadModel(model)

        # self.dispatcher.dispatch(KernelDispatcher.POST_CREATE) post-create hook.
