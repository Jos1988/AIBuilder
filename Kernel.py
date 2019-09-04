from typing import List

import numpy as np

from AIBuilder.AI import AI
from AIBuilder.AIFactory.AIFactory import AIFactory
from AIBuilder.AIFactory.Builders import Builder
from AIBuilder.AITester import AITester, GainBasedFeatureImportance, AccuracyBaselineDiff, Predictor
from AIBuilder.Dispatching import KernelDispatcher, FactoryEvent, TesterEvent, ModelNotUniqueObserver, \
    PreCreateObserver, \
    PreTrainObserver, PostTrainObserver, PreEvaluationObserver, PostEvaluationObserver, Session, PreRunObserver, \
    PostRunObserver, PreRunLogObserver, PostEvaluationLogObserver, PreRunLoadMetaLogger, PreRunLoadSummaryLogger
from AIBuilder.Summizer import TimeSummizer


class Kernel:
    model: AI
    all_models = List[AI]

    def __init__(self, session: Session, **kwargs):
        self.all_models = []
        self.model = None
        self.session = session
        self.factory = None
        self.tester = None
        self.dispatcher = None
        self.no_log = False
        self.prediction_results = None

        if 'no_log' in kwargs:
            self.no_log = kwargs['no_log']

    def boot(self, train: bool = True, evaluate: bool = True, predict: bool = False, cache=True):
        self.train = train
        self.evaluate = evaluate
        self.predict = predict

        self._load_dispatcher()

        # self.dispatcher.dispatch(KernelDispatcher.PRE_BOOT) pre-boot hook
        self.factory = AIFactory(
            builders=Builder.get_all_registered(),
            project_name=self.session.project_name,
            log_dir=self.session.log_dir,
            dispatcher=self.dispatcher,
            cache=cache
        )
        summizer = TimeSummizer()

        feature_importance = GainBasedFeatureImportance()
        accuracy_diff = AccuracyBaselineDiff()

        self.tester = AITester(summizer=summizer, evaluators=[feature_importance, accuracy_diff])

        if self.predict:
            self.predictor = Predictor(categories=self.session.session_data['prediction']['categories'])

        # self.dispatcher.dispatch(KernelDispatcher.POST_BOOT) post-boot hook

    def _load_dispatcher(self):
        self.dispatcher = KernelDispatcher()
        self.dispatcher.addObserver(PreRunObserver())
        self.dispatcher.addObserver(ModelNotUniqueObserver())
        self.dispatcher.addObserver(PreCreateObserver())
        if self.train:
            self.dispatcher.addObserver(PreTrainObserver())
            self.dispatcher.addObserver(PostTrainObserver())
        if self.evaluate:
            self.dispatcher.addObserver(PreEvaluationObserver())
            self.dispatcher.addObserver(PostEvaluationObserver())
        self.dispatcher.addObserver(PostRunObserver())
        if True is not self.no_log and self.evaluate:
            self.dispatcher.addObserver(PreRunLoadMetaLogger())
            self.dispatcher.addObserver(PreRunLoadSummaryLogger())
            self.dispatcher.addObserver(PostEvaluationLogObserver())

    def run(self):
        self.model = None
        pre_run_event = TesterEvent(event_name=KernelDispatcher.PRE_RUN, session=self.session, tester=self.tester)
        self.dispatcher.dispatch(pre_run_event)
        while self.factory.has_next_ai():
            #todo: move the new is unique logic somewhere else and changing sequence of operation seems to difficult, smell?
            final_description = self.preview_model_description()
            final_description_hash = AITester.stable_hash_description(self.preview_model_description())
            log_dir_path = self.factory.log_dir
            if not AITester.is_hash_unique_to_report(log_dir_path=log_dir_path, description_hash=final_description_hash):
                self.factory.builder_permutations.pop()
                #todo: some kind of null model, as it is just used to pervey name and log dir?
                model = AI(self.factory.project_name, self.factory.log_dir, self.factory.project_name + '_X')
                model.description = final_description
                self.tester.ml_model = model
                self.ModelNotUnique()
                continue

            self.doCreateModel()

            if self.train:
                self.doTrainModel()

            self.all_models.append(self.model)
            if self.evaluate:
                self.doEvaluateModel()

        if self.predict:
            self.prediction_results = self.doPredict()

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

    def preview_model_description(self) -> dict:
        return self.factory.preview_final_description()

    def doCreateModel(self):
        pre_create_event = FactoryEvent(event_name=KernelDispatcher.PRE_CREATE, session=self.session,
                                        factory=self.factory)
        self.dispatcher.dispatch(pre_create_event)

        model = self.factory.create_next_ai()
        self.model = model
        self.tester.loadModel(model)

        # self.dispatcher.dispatch(KernelDispatcher.POST_CREATE) post-create hook.

    def doPredict(self):
        print('=======================================')
        print('start prediction')
        print('=======================================')
        results = self.predictor.predict(self.all_models)
        print('=======================================')
        print('finished prediction')
        print('=======================================')
        # dispatch post predict event

        return results
