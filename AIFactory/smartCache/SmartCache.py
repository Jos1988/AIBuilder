import functools
from copy import deepcopy
from typing import List, Optional


class Instruction:
    """ Instruction passed to cache, tells smart cache what to do.

    Instructions:
        WARM_CACHE: store output in cache
        LOAD_CACHE: use output stored in cache
        NO_EXECUTION: do nothing, both caching and the decorated function wont be used.
        NO_CACHE: operate decorated function without caching.

    """
    # INSTRUCTIONS
    WARM_CACHE = 'warm'
    LOAD_CACHE = 'load'
    FUNCTION_CACHE = 'function_cache'
    NO_EXECUTION = 'no execution'
    NO_CACHE = 'no cache'

    valid_instructions = [WARM_CACHE, LOAD_CACHE, NO_EXECUTION, NO_CACHE, FUNCTION_CACHE]

    def __init__(self, instruction: str, instruction_condition: callable, payload=None):
        """
        Args:
            instruction: Valid instruction.
            instruction_condition: callable that takes the call number (int) and returns whether this instructions is
            valid for the respective call number (bool)
        """
        if instruction not in self.valid_instructions:
            raise RuntimeError(f'"{instruction}" is not a valid instruction.')

        self.instruction_condition = instruction_condition
        self.instruction = instruction
        self.payload = payload
        self.times_called = 0

    def validate_call_count(self, call_count: int) -> bool:
        return self.instruction_condition(call_count)


class InstructionSet:

    def __init__(self, *instructions: Instruction):
        self.instructions = instructions

    def get_instruction(self, call_count: int) -> Instruction:
        """ Filter retrieve correct instruction from set.

        Args:
            call_count: previous calls to the specified object and method, zero indexed.
        """
        valid_instructions = [instruction for instruction in self.instructions
                              if instruction.validate_call_count(call_count)]

        if len(valid_instructions) is 1:
            return valid_instructions[0]

        raise LookupError(
            f'found invalid number of instructions ({len(valid_instructions)}) for call count {call_count}.')


class MethodCall:
    def __init__(self, obj_hash: int, method_name: str):
        self.method_name = method_name
        self.obj_hash = obj_hash
        self.call_count = None
        self.instructions = None

    @staticmethod
    def create(obj: object,
               method_name: str,
               call_count: Optional[int] = None,
               instructions: Optional[InstructionSet] = None):

        call = MethodCall(obj.__hash__(), method_name=method_name)
        if call_count is not None:
            call.call_count = call_count

        if instructions is not None:
            call.instructions = instructions

        return call


def check_one_or_none_result(search_result: List[tuple]) -> Optional[tuple]:
    result_count = len(search_result)
    if result_count is 1:
        return search_result[0]
    elif result_count is 0:
        return None

    raise LookupError(f'Found not 1 but {result_count} results.')


class CallCountLog:

    def __init__(self):
        self.call_log = []

    def log_method_call(self, call: MethodCall):
        record = self._lookup_method_call_record(call)
        count = 1
        if record is not None:
            count = record[2]
            count += 1
            self.call_log.remove(record)

        self.call_log.append((call.obj_hash, call.method_name, count))

    def set_method_call_count(self, call: MethodCall):
        record = self._lookup_method_call_record(call)

        if record is None:
            call.call_count = 0
            return

        call.call_count = record[2]

    def _lookup_method_call_record(self, call: MethodCall) -> Optional[tuple]:
        search_result = [(hash, method, count) for hash, method, count in self.call_log
                         if hash == call.obj_hash and method == call.method_name]

        if len(search_result) is 1:
            return search_result[0]
        if len(search_result) is 0:
            return None

        raise LookupError(f'An error occurd while looking up hash: {call.obj_hash}, '
                          f'and method_name: {call.method_name}, result {search_result} found.')


class InstructionsRepository:

    def __init__(self):
        self.repository = []

    def has_instruction_set(self, call: MethodCall) -> bool:
        if None is self._lookup_record(call):
            return False

        return True

    def remove_instruction_set(self, call: MethodCall):
        if not self.has_instruction_set(call):
            return

        record = self._lookup_record(call)
        self.repository.remove(record)

    def add_instruction_set(self, call: MethodCall):
        assert None is not call.instructions and type(call.instructions) is InstructionSet, \
            f'cannot set instructions to {call.instructions}'
        if self.has_instruction_set(call):
            raise RuntimeError('instructions already set.')

        record = (call.obj_hash, call.method_name, call.instructions)
        self.repository.append(record)

    def get_instruction_set(self, call: MethodCall) -> Optional[InstructionSet]:
        record = self._lookup_record(call)
        if None is record:
            raise RuntimeError('Instructions not found.')

        instruction_set: InstructionSet = record[2]

        return instruction_set

    @staticmethod
    def _validate_method_call(call: MethodCall):
        assert None is not call.obj_hash
        assert None is not call.method_name

    def _lookup_record(self, call: MethodCall) -> Optional[tuple]:
        self._validate_method_call(call)
        search_result = [(hash, method, _) for hash, method, _ in self.repository if
                         hash == call.obj_hash and method == call.method_name]

        try:
            return check_one_or_none_result(search_result)
        except:
            raise LookupError(f'An error occurd while looking up hash: {call.obj_hash}, '
                              f'and method_name: {call.obj_hash}, result {search_result} found.')


class SmartCacheManager:

    def __init__(self, instruction_repo: InstructionsRepository, call_log: CallCountLog, verbosity: int = 0):
        self.instruction_repo = instruction_repo
        self.call_log = call_log
        self.verbosity = verbosity
        self.cache = {}

    def has_instruction(self, obj: object, method: str) -> bool:
        call = MethodCall.create(obj, method)
        return self.instruction_repo.has_instruction_set(call)

    def get_instruction(self, obj: object, method: str) -> Instruction:
        call = MethodCall.create(obj, method)
        self.call_log.set_method_call_count(call)
        instruction_set = self.instruction_repo.get_instruction_set(call)
        instruction = instruction_set.get_instruction(call.call_count)
        self.call_log.log_method_call(call)

        return instruction

    def add_request_instructions(self, obj: object, method: str, instructions: InstructionSet):
        call = MethodCall.create(obj, method, instructions=instructions)
        self.instruction_repo.add_instruction_set(call)

    def set_cache(self, obj: object, method: str, method_output):
        call = MethodCall.create(obj, method)
        self.cache[str(call.obj_hash) + method] = deepcopy(method_output)

    def load_from_cache(self, obj: object, method: str):
        call = MethodCall.create(obj, method)
        cached_data = self.cache[str(call.obj_hash) + method]
        return deepcopy(cached_data)


smart_cache_manager = SmartCacheManager(instruction_repo=InstructionsRepository(), call_log=CallCountLog(), verbosity=1)


def smart_cache(fn):

    function_cache = {}

    @functools.wraps(fn)
    def fn_wrapper(*args, **kwargs):
        if False is isinstance(smart_cache_manager, SmartCacheManager):
            return fn(*args, **kwargs)

        if smart_cache_manager.verbosity > 1:
            print('smart cache: running smart cache wrapper')
            print(f'smart cache: {args[0].__class__}.{fn.__name__}')

        if False is smart_cache_manager.has_instruction(args[0], fn.__name__):
            if smart_cache_manager.verbosity > 1:
                print('smart cache: no instructions')
            return fn(*args, **kwargs)

        instruction = smart_cache_manager.get_instruction(args[0], fn.__name__)
        # instruction = instruction.instruction
        if instruction.instruction is Instruction.NO_EXECUTION:
            if smart_cache_manager.verbosity > 0:
                print('smart cache: skipping.')
            return None

        if instruction.instruction is Instruction.NO_CACHE:
            if smart_cache_manager.verbosity > 1:
                print('smart cache: no caching.')
            return fn(*args, **kwargs)

        if instruction.instruction is Instruction.FUNCTION_CACHE:
            if smart_cache_manager.verbosity > 0:
                print('smart cache: function cache')

            if 'argument_hash_fn' not in instruction.payload:
                raise RuntimeError(f'function cache requires argument_hash_fn in payload.')

            argument_hash_fn = instruction.payload['argument_hash_fn']
            key = argument_hash_fn(*args, **kwargs)
            if key in function_cache:
                if smart_cache_manager.verbosity > 1:
                    print(f'load {key} from fn cache.')

                return deepcopy(function_cache[key])

            output = fn(*args, **kwargs)
            if output is None:
                raise RuntimeError('"None" output encountered, not a valid caching value.')

            function_cache[key] = deepcopy(output)
            return output

        if instruction.instruction is Instruction.WARM_CACHE:
            if smart_cache_manager.verbosity > 0:
                print('smart cache: warms cache')
            output = fn(*args, **kwargs)

            smart_cache_manager.set_cache(args[0], fn.__name__, method_output=output)
            return output

        if instruction.instruction is Instruction.LOAD_CACHE:
            if smart_cache_manager.verbosity > 0:
                print('smart cache: load output from cache.')
            return smart_cache_manager.load_from_cache(args[0], fn.__name__)

        raise RuntimeError(f'Instruction "{instruction.instruction}" not recognised.')

    return fn_wrapper
