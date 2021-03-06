from typing import List, Dict, Any
from itertools import chain
from AIBuilder.AI import AI, AbstractAI
from AIBuilder.AIFactory.Builders import Builder, EstimatorBuilder
from AIBuilder.AIFactory.Printing import ConsolePrintStrategy, FactoryPrinter
from AIBuilder.AIFactory.smartCache.SmartCache import SmartCacheManager, InstructionSet, Instruction, \
    smart_cache_manager
from AIBuilder.Dispatching import Dispatcher, PreBuilderObserver, ModelEvent, KernelDispatcher


class BuilderSorter(object):
    builders_sorted = list
    loaded_builders = list
    builders_by_name = dict
    unloaded_builders = list

    def clear(self):
        self.builders_sorted = []
        self.loaded_builders = []
        self.builders_by_name = {}
        self.unloaded_builders = []

    def sort_permutations(self, builders: List[List[Builder]]) -> List[List[Builder]]:
        sorted_permutations = []
        for permutation in builders:
            sorted_permutation = self.sort(builders=permutation)
            sorted_permutations.append(sorted_permutation)

        return sorted_permutations

    def sort(self, builders: List[Builder]) -> List[Builder]:
        self.clear()

        for builder in builders:
            self.unloaded_builders.append(builder)
            self.builders_by_name[builder.builder_type] = builder

        while len(self.unloaded_builders) is not 0:
            builder = self.unloaded_builders.pop()
            if not self.has_unloaded_dependencies(builder):
                self.load_builder(builder)
                continue

            self.unloaded_builders.append(builder)

            dependency = self.get_next_loadable_dependency(builder)
            self.unloaded_builders.remove(dependency)
            self.load_builder(dependency)

        return self.builders_sorted

    def has_unloaded_dependencies(self, builder: Builder):
        dependencies = builder.dependent_on
        if len(dependencies) is 0:
            return False

        for dependency in dependencies:
            if dependency not in self.loaded_builders:
                return True

        return False

    def load_builder(self, builder: Builder):
        self.loaded_builders.append(builder.builder_type)
        if builder in self.unloaded_builders:
            self.unloaded_builders.remove(builder)

        self.builders_sorted.append(builder)

    def get_next_loadable_dependency(self, builder: Builder) -> Builder:
        dependencies = builder.dependent_on

        if len(dependencies) == 0:
            raise RuntimeError('{} has no dependencies, so cannot get next loadable dependency.'
                               .format(builder.__class__.__name__))

        for dependency in dependencies:
            assert dependency in self.builders_by_name, '{} is missing a dependency: {}' \
                .format(builder.__class__.__name__, dependency)

            dependent_builder = self.builders_by_name[dependency]

            if self.has_unloaded_dependencies(dependent_builder):
                return self.get_next_loadable_dependency(dependent_builder)

            if dependent_builder in self.unloaded_builders:
                return dependent_builder

            continue


class PermutationGenerator(object):
    permutations: List[List[Any]]
    builder_types: List[str]
    grouped_builders: Dict[str, List[Builder]]
    all_builders: List[Builder]

    def generate(self, builders: List[Builder]) -> list:
        self.all_builders = builders
        self.group_builders()
        self.walk_layers()

        return self.permutations

    def group_builders(self):
        self.grouped_builders = {}
        self.builder_types = []
        for builder in self.all_builders:
            if builder.builder_type not in self.grouped_builders:
                self.grouped_builders[builder.builder_type] = []
                self.builder_types.append(builder.builder_type)

            self.grouped_builders[builder.builder_type].append(builder)

    def walk_layers(self):
        layers = self.builder_types
        self.permutations = [[]]
        for layer in layers:
            self.permutations = self.walk_layer_with_permutations(layer, permutations=self.permutations)

    def walk_layer_with_permutations(self, layer_name: str, permutations: List[list]):
        layer_options = self.grouped_builders[layer_name]

        new_permutations = []
        for permutation in permutations:
            for option in layer_options:
                new_permutations.append(permutation + [option])

        return new_permutations


class BuilderInstructionModel:
    """ Used for loading instructions for the Builders

    Props
        builder: respective Builder object
        prev_builders: description of builders that have ran before and including the current builder. In essence a
            description of the whole building process up to and including the current builder.
        builder_instructions: InstructionSet for builder.

    """

    def __init__(self, builder: Builder):
        self.builder = builder
        self.prev_builders = None
        self.builder_instructions = None


class CachingInstructionsLoader:
    """ Provides functionality for caching the builder process.

        permutations:
            Builders are received in sorted order. Each builder can be used in combination with varying other builders.
            Each series of builders that are run, is considered a permutation of builder.
            for example when whe have c builders: A, B, C1 and C2. Where C1 and C2 are interchangeable, there are two
            possible permutations:
            permutation 1: [BuilderA, BuilderB, BuilderC]
            permutation 2: [BuilderA, BuilderB, BuilderD]

        prev_builders:
            The previous builder description is a string consisting of the hashes of the builders up to and
            including the current builder. If two of these descriptions are duplicate it means that these builders
            will be used subsequently on multiple occasions, so caching is possible.

        Example:
        The following builders will be run in the following orders.
            permutation 1 [BuilderA, BuilderB, BuilderC1]
            permutation 2 [BuilderA, BuilderB, BuilderC2]

        The builders carry the following hash descriptions;
            BuilderA: 111
            BuilderB: 222
            BuilderC1: 333
            BuilderC2: 444

        This will result in the following 'previous builder' descriptions:
            permutation 1 [111, 111222, 111222333]
            permutation 2 [111, 111222, 111222444]

        As the builder descriptions 111 and 111222 are duplicate these processes should be cached. The first iteration
        of the duplicate series should warm the cache, the second iteration should use the cache, only at the last
        builder.
            permutation 1 [RUN NORMALLY, USE FUNCTION CACHE, RUN NORMALLY]
            permutation 2 [SKIP, USE FUNCTION CACHE, RUN NORMALLY]

    """

    SKIP_BUILDING = 'skip_building'
    USE_CACHE = 'use_cache'
    NO_ACTION = 'no_action'

    # Do not cache estimator builder as it must assign a unique name to the estimator every time.
    BLACK_LIST = [EstimatorBuilder]

    def __init__(self, manager: SmartCacheManager):
        self.manager = manager
        self.all_prev_builder_combinations = []

    def set_caching_instructions(self, builders: List[List[Builder]]):
        """ Creates instructions for each step in all the builder permutations.
            These instructions tell the factory whether to:
                - run the builder normally;
                - skip the builder;
                - warm-up the cache;
                - use the cache.
        """
        models = self.load_builder_models(builders)
        self.load_prev_builder_descriptions(models)
        self.map_instructions_to_models(models)
        models = chain.from_iterable(models)
        loaded_builders = []
        for model in models:
            if model.builder not in loaded_builders:
                self.manager.add_request_instructions(model.builder, 'build', model.builder_instructions)
                loaded_builders.append(model.builder)

    def load_builder_models(self, permutations: List[List[Builder]]) -> List[List[BuilderInstructionModel]]:
        """ Transforms a 2d array of Builders in to 2d array of BuilderInstructionModels. """
        return [[BuilderInstructionModel(builder) for builder in permutation] for permutation in permutations]

    def load_prev_builder_descriptions(self, permutations: List[List[BuilderInstructionModel]]):
        """ Load prev_builder descriptions into the sorting models to prepare them for analysis. """
        for permutation in permutations:
            prev_builders = ''
            for builderInstructionModel in permutation:
                prev_builders += str(builderInstructionModel.builder.__hash__())
                builderInstructionModel.prev_builders = prev_builders

    def map_instructions_to_models(self, models: List[List[BuilderInstructionModel]]):
        """ Load instructions for caching and stores them by builder models. """
        self._load_all_previous_builders(models)
        self._handle_duplicate_series(models)
        self._handle_remaining_models(models)

    def _handle_duplicate_series(self, models):
        """ Finds al series of builder that occur than once and instructs caching behavior. """
        for permutation in models:
            duplicate_series_of_builders = self._get_duplicate_series_of_builders(permutation)
            if len(duplicate_series_of_builders) > 0:
                self._instruct_duplicate_serie_of_builders(duplicate_series_of_builders)

    def _get_duplicate_series_of_builders(self, permutation: List[BuilderInstructionModel]) \
            -> List[BuilderInstructionModel]:
        """ returns a list of builder that will be run multiple times. """
        duplicate_series_of_builders = []
        for model in permutation:
            if self._is_do_not_cache(model.builder):
                return duplicate_series_of_builders

            if False is self._previous_builders_unique(model):
                duplicate_series_of_builders.append(model)

        return duplicate_series_of_builders

    def _load_all_previous_builders(self, models: List[List[BuilderInstructionModel]]):
        """ Loads all previous builder information into a list, which is set as a class attribute """
        for permutation in models:
            for model in permutation:
                self.all_prev_builder_combinations.append(model.prev_builders)

    def _instruct_duplicate_serie_of_builders(self, series_of_builders: List[BuilderInstructionModel]):
        """ Handle a series of duplicate builders, the last one will receive instructions to cache. All the others
         will receive instructions to run once, than skip.
        """

        def only_first_iteration(iteration: int) -> bool:
            return iteration == 0

        def after_first_iteration(iteration: int) -> bool:
            return iteration > 0

        def always(iteration: int) -> bool:
            return True

        def hash_args(builder: Builder, model: AbstractAI):
            return hash(model)

        for model in series_of_builders:
            payload = {'argument_hash_fn': hash_args}
            cache_output = InstructionSet(Instruction(Instruction.FUNCTION_CACHE, always, payload))
            model.builder_instructions = cache_output

    def _previous_builders_unique(self, model: BuilderInstructionModel) -> bool:
        """ Check if the sequence of builders that have ran up to and including this builder is unique. """
        return self.all_prev_builder_combinations.count(model.prev_builders) <= 1

    def _handle_remaining_models(self, models: List[List[BuilderInstructionModel]]):
        for permutation in models:
            for model in permutation:
                if None is model.builder_instructions:
                    self._instruct_normal_behaviour(model)

    def _instruct_normal_behaviour(self, model: BuilderInstructionModel):
        def always(iteration: int) -> bool:
            return True

        normal_behaviour = InstructionSet(Instruction(Instruction.NO_CACHE, always))
        model.builder_instructions = normal_behaviour

    def _is_do_not_cache(self, builder: Builder) -> bool:
        for blacklisted_class in self.BLACK_LIST:
            if isinstance(builder, blacklisted_class):
                return True

        return False


class AIFactory:
    builder_permutations: List[List[Builder]]

    def __init__(self, builders: List[Builder], project_name: str, log_dir: str, dispatcher: Dispatcher,
                 cache: bool = True):
        self.console_printer = FactoryPrinter(ConsolePrintStrategy())
        self.sorter = BuilderSorter()
        self.project_name = project_name
        self.log_dir = log_dir
        self.dispatcher = dispatcher
        self.permutation_generator = PermutationGenerator()
        self.caching_instruction_loader = CachingInstructionsLoader(manager=smart_cache_manager)

        self._add_observers()
        self.builder_permutations = self.load_permutations(builders)
        if cache:
            self.caching_instruction_loader.set_caching_instructions(self.builder_permutations)

    def load_permutations(self, builders):
        builder_permutations = self.permutation_generator.generate(builders=builders)
        return self.sorter.sort_permutations(builder_permutations)

    def _add_observers(self):
        self.dispatcher.addObserver(PreBuilderObserver())

    def count_remaining_models(self):
        return len(self.builder_permutations)

    def has_next_ai(self):
        return self.count_remaining_models() is not 0

    def preview_final_description(self) -> dict:
        next_builder_permutation = self.builder_permutations[-1:][0]
        final_description = {}
        for builder in next_builder_permutation:
            final_description[builder.builder_type] = builder.describe()

        return final_description

    def create_next_ai(self):
        next_builder_permutation = self.builder_permutations.pop()

        return self.create_AI(next_builder_permutation)

    def create_AI(self, builders: list, ai_name: str = None) -> AbstractAI:
        self.validate_builders(builders)
        ml_model = AI(self.project_name, self.log_dir, ai_name)
        self.dispatcher.dispatch(ModelEvent(KernelDispatcher.PRE_RUN_BUILDERS, ml_model))
        name = ml_model.get_name()

        description = {}
        for builder in builders:
            self.console_printer.line('running: ' + builder.__class__.__name__)
            if isinstance(builder, EstimatorBuilder):
                # overwrite old name from being loaded from cache.
                ml_model.set_name(name)

            result = builder.build(ml_model)
            if result is not None:
                # Result will be None if cache prevents execution, keep using the old model as it has the description,
                # which is required for determining function cache key.
                # todo: fix this, we should not be passing a model just because the cache needs the description.
                ml_model = result

            description[builder.builder_type] = builder.describe()
            ml_model.description = description

        return ml_model

    @staticmethod
    def validate_builders(builders: list):
        for builder in builders:
            builder.validate()
