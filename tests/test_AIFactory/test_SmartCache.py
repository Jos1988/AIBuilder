import unittest

from AIBuilder.AIFactory.smartCache.SmartCache import smart_cache, SmartCacheManager, InstructionSet, \
    Instruction, InstructionsRepository, CallCountLog, MethodCall, smart_cache_manager


class TestBuilderCache(unittest.TestCase):

    def setUp(self) -> None:
        self.set_A = InstructionSet(
            Instruction('NO_CACHE', lambda i: i is 0),
            Instruction('SKIP_OPERATION', lambda i: i > 0)
        )

        self.set_B = InstructionSet(
            Instruction('SAVE_CACHE', lambda i: i is 0),
            Instruction('LOAD_CACHE', lambda i: i > 0),
        )

    def test_instructions(self):
        instruction_sequence = []
        expected_instruction_sequence = ['NO_CACHE', 'SAVE_CACHE', 'SKIP_OPERATION', 'LOAD_CACHE',
                                         'SKIP_OPERATION', 'LOAD_CACHE', 'SKIP_OPERATION', 'LOAD_CACHE',
                                         'SKIP_OPERATION', 'LOAD_CACHE']

        for i in range(5):
            for instruction_set in [self.set_A, self.set_B]:
                instruction_sequence.append(instruction_set.get_instruction(i))

        self.assertEqual(instruction_sequence, expected_instruction_sequence)


class TestCacheDecorator(unittest.TestCase):

    def setUp(self):
        self.something_calculator1 = SomethingCalculatorTestClass()

    def test_decorator(self):
        print(self.something_calculator1)
        result = self.something_calculator1.return_something('ignore')
        print(result)


class TestCallCountLog(unittest.TestCase):

    def setUp(self) -> None:
        self.log = CallCountLog()

    def test_logging(self):
        calls = [
            (1, 'a'),
            (1, 'b'),
            (2, 'a'),
            (2, 'b'),
            (1, 'b'),
            (1, 'b'),
            (3, 'c'),
        ]

        for object_hash, method_name in calls:
            call = MethodCall(object_hash, method_name)
            self.log.log_method_call(call)

        calls_to_check = [(MethodCall(1, 'a'), 1), (MethodCall(2, 'a'), 1), (MethodCall(1, 'b'), 3),
                          (MethodCall(2, 'b'), 1), (MethodCall(3, 'a'), 0), (MethodCall(3, 'c'), 1)]

        for call, expected_count in calls_to_check:
            self.log.set_method_call_count(call)
            self.assertEqual(call.call_count, expected_count)


class TestInstructionsRepository(unittest.TestCase):

    def setUp(self) -> None:
        self.repository = InstructionsRepository()

    def test_loading_retrieving_removing(self):
        call_to_obj_1_method_A = MethodCall(obj_hash=1, method_name='A')
        call_to_obj_1_method_A.call_count = 1
        call_to_obj_1_method_A.instructions = InstructionSet(Instruction('instruction1', lambda x: x > 1))

        call_to_obj_1_method_B = MethodCall(obj_hash=1, method_name='B')
        call_to_obj_1_method_B.call_count = 2
        call_to_obj_1_method_B.instructions = InstructionSet(Instruction('instruction2', lambda x: x > 1))

        call_to_obj_2_method_A = MethodCall(obj_hash=2, method_name='A')
        call_to_obj_2_method_A.call_count = 3
        call_to_obj_2_method_A.instructions = InstructionSet(Instruction('instruction3', lambda x: x > 1))

        call_to_obj_2_method_B = MethodCall(obj_hash=2, method_name='B')
        call_to_obj_2_method_B.call_count = 4
        call_to_obj_2_method_B.instructions = InstructionSet(Instruction('instruction4', lambda x: x > 1))

        self.assertFalse(self.repository.has_instruction_set(call_to_obj_1_method_A))
        self.assertFalse(self.repository.has_instruction_set(call_to_obj_1_method_B))
        self.assertFalse(self.repository.has_instruction_set(call_to_obj_2_method_A))
        self.assertFalse(self.repository.has_instruction_set(call_to_obj_2_method_B))

        self.repository.add_instruction_set(call_to_obj_1_method_A)
        self.repository.add_instruction_set(call_to_obj_1_method_B)
        self.repository.add_instruction_set(call_to_obj_2_method_A)
        self.repository.add_instruction_set(call_to_obj_2_method_B)

        self.assertTrue(self.repository.has_instruction_set(call_to_obj_1_method_A))
        self.assertTrue(self.repository.has_instruction_set(call_to_obj_1_method_B))
        self.assertTrue(self.repository.has_instruction_set(call_to_obj_2_method_A))
        self.assertTrue(self.repository.has_instruction_set(call_to_obj_2_method_B))

        instruction1 = self.repository.get_instruction_set(MethodCall(obj_hash=1, method_name='A'))
        instruction2 = self.repository.get_instruction_set(MethodCall(obj_hash=1, method_name='B'))
        instruction3 = self.repository.get_instruction_set(MethodCall(obj_hash=2, method_name='A'))
        instruction4 = self.repository.get_instruction_set(MethodCall(obj_hash=2, method_name='B'))

        self.assertEqual('instruction1', instruction1.get_instruction(2))
        self.assertEqual('instruction2', instruction2.get_instruction(2))
        self.assertEqual('instruction3', instruction3.get_instruction(2))
        self.assertEqual('instruction4', instruction4.get_instruction(2))

        self.repository.remove_instruction_set(MethodCall(obj_hash=1, method_name='A'))
        self.repository.remove_instruction_set(MethodCall(obj_hash=1, method_name='B'))
        self.repository.remove_instruction_set(MethodCall(obj_hash=2, method_name='A'))
        self.repository.remove_instruction_set(MethodCall(obj_hash=2, method_name='B'))

        self.assertFalse(self.repository.has_instruction_set(call_to_obj_1_method_A))
        self.assertFalse(self.repository.has_instruction_set(call_to_obj_1_method_B))
        self.assertFalse(self.repository.has_instruction_set(call_to_obj_2_method_A))
        self.assertFalse(self.repository.has_instruction_set(call_to_obj_2_method_B))

    def test_removing_non_existing_call(self):
        self.repository.remove_instruction_set(MethodCall(obj_hash=1, method_name='A'))

    def test_retrieve_non_existing_call(self):
        with self.assertRaises(RuntimeError, msg='Instructions not found.'):
            self.repository.get_instruction_set(MethodCall(obj_hash=1, method_name='A'))

    def test_instruction_set_twice(self):
        method_call = MethodCall(obj_hash=2, method_name='B')
        method_call.call_count = 4
        method_call.instructions = InstructionSet(Instruction('instruction4', lambda x: x > 1))
        self.repository.add_instruction_set(method_call)
        with self.assertRaises(RuntimeError, msg='Instructions already set.'):
            self.repository.add_instruction_set(method_call)


class testObject:
    pass


class TestSmartCacheManager(unittest.TestCase):

    def setUp(self) -> None:
        self.instructions_repo = InstructionsRepository()
        self.call_count_log = CallCountLog()
        self.manager = SmartCacheManager(self.instructions_repo, self.call_count_log)

        obj_1_call_to_A = InstructionSet(Instruction('FIRST', lambda i: i is 0), Instruction('OTHER', lambda i: i > 0))
        obj_1_call_to_B = InstructionSet(Instruction('FIRST', lambda i: i is 0), Instruction('OTHER', lambda i: i > 0))
        obj_2_call_to_A = InstructionSet(Instruction('FIRST', lambda i: i is 0), Instruction('OTHER', lambda i: i > 0))
        obj_2_call_to_B = InstructionSet(Instruction('FIRST', lambda i: i is 0), Instruction('OTHER', lambda i: i > 0))

        self.o1 = testObject()
        self.o2 = testObject()
        self.manager.add_request_instructions(obj=self.o1, method='A', instructions=obj_1_call_to_A)
        self.manager.add_request_instructions(obj=self.o1, method='B', instructions=obj_1_call_to_B)
        self.manager.add_request_instructions(obj=self.o2, method='A', instructions=obj_2_call_to_A)
        self.manager.add_request_instructions(obj=self.o2, method='B', instructions=obj_2_call_to_B)

    def test_get_instruction(self):
        expected_sequence = ['FIRST', 'FIRST', 'FIRST', 'FIRST', 'OTHER', 'OTHER', 'OTHER', 'OTHER']
        created_sequence = []
        for obj, method_name in [(self.o1, 'A'), (self.o1, 'B'), (self.o2, 'A'), (self.o2, 'B'), (self.o1, 'A'),
                                 (self.o1, 'B'), (self.o2, 'A'), (self.o2, 'B')]:
            instruction = self.manager.get_instruction(obj=obj, method=method_name)
            created_sequence.append(instruction)
            self.call_count_log.log_method_call(MethodCall.create(obj=obj, method_name=method_name))

        self.assertEqual(expected_sequence, created_sequence)


class SomethingCalculatorTestClass:

    def __init__(self):
        self.fn_called = 0

    @smart_cache
    def return_something(self, input: int):
        self.fn_called += 1
        output = input + 1

        return output


class TestCaching(unittest.TestCase):

    def setUp(self) -> None:
        instructions = InstructionSet(Instruction(Instruction.WARM_CACHE, lambda x: x == 0),
                                      Instruction(Instruction.NO_CACHE, lambda x: x == 1),
                                      Instruction(Instruction.NO_EXECUTION, lambda x: x == 2),
                                      Instruction(Instruction.LOAD_CACHE, lambda x: x > 2))
        self.subject = SomethingCalculatorTestClass()
        smart_cache_manager.add_request_instructions(self.subject, 'return_something', instructions=instructions)

    def test_cache(self):
        self.assertEqual(2, self.subject.return_something(1))
        self.assertEqual(4, self.subject.return_something(3))
        self.assertEqual(None, self.subject.return_something(1))
        self.assertEqual(2, self.subject.return_something(8))
        self.assertEqual(2, self.subject.return_something(9))
        self.assertEqual(2, self.subject.return_something(10))
        self.assertEqual(2, self.subject.fn_called)


class TestFunctionCaching(unittest.TestCase):

    def setUp(self) -> None:

        def argument_hash_fn(*args, **kwargs):
            return (args[1:],) + tuple(sorted(kwargs.items()))

        payload = {'argument_hash_fn': argument_hash_fn}

        instructions = InstructionSet(Instruction(Instruction.FUNCTION_CACHE, lambda x: x == 0, payload),
                                      Instruction(Instruction.FUNCTION_CACHE, lambda x: x > 0, payload))

        self.subject = SomethingCalculatorTestClass()
        smart_cache_manager.add_request_instructions(self.subject, 'return_something', instructions=instructions)

    def test_cache(self):
        self.assertEqual(1, self.subject.return_something(0))
        self.assertEqual(3, self.subject.return_something(2))
        self.assertEqual(1, self.subject.return_something(0))
        self.assertEqual(3, self.subject.return_something(2))
        self.assertEqual(4, self.subject.return_something(3))
        self.assertEqual(4, self.subject.return_something(3))
        self.assertEqual(3, self.subject.fn_called)
