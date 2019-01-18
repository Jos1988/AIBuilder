import unittest

from AIBuilder.category_deduplication import StringDeduplicator, AliasListBuilder


class TestAliasListBuilder(unittest.TestCase):

    def setUp(self):
        self.groups = [{'bird2', 'bird'}, {'cat2', 'cat', 'cat3', 'cat1'}, {'doggg', 'dog', 'Dog2', 'DOG2', 'dogg'}]
        self.expected = {'DOG2': 'dog', 'Dog2': 'dog', 'bird2': 'bird', 'cat1': 'cat', 'cat2': 'cat', 'cat3': 'cat',
                         'dogg': 'dog', 'doggg': 'dog'}
        self.alias_list_builder = AliasListBuilder()

    def test_convert_groups(self):
        result = self.alias_list_builder.convert_groups(self.groups)
        for exp_key, exp_value in self.expected.items():
            self.assertEqual(result[exp_key], exp_value)

        self.assertEqual(len(self.expected), len(result))


class TestStringDeduplicator(unittest.TestCase):

    def setUp(self):
        self.data_small = ['cat', 'dog', 'bird', 'Cat']
        self.data_large = ['cat', 'dog', 'bird', 'cat1', 'cat2', 'cat3', 'lizard', 'Dog', 'dogg', 'doggg', 'DOG',
                           'bird2', 'cow',
                           'sheep', 'chicken', 'goose', 'peacock', 'pig', 'pork', 'goat']

    def test_deduplicate_small(self):
        deduplicator = StringDeduplicator(self.data_small, 0.8)
        deduplicator.deduplicate()
        self.assertListEqual(deduplicator.deduplicated_list, ['Cat', 'bird', 'dog'])
        self.assertEqual(deduplicator.duplicate_groups, [{'Cat', 'cat'}])

    def test_deduplicate_large(self):
        deduplicator = StringDeduplicator(self.data_large, 0.8)
        deduplicator.deduplicate()
        self.assertEqual(deduplicator.duplicate_groups,
                         [{'bird2', 'bird'}, {'cat2', 'cat', 'cat3', 'cat1'}, {'doggg', 'dog', 'Dog', 'DOG', 'dogg'}])
        self.assertEqual(deduplicator.deduplicated_list,
                         ['goat', 'pork', 'pig', 'peacock', 'goose', 'chicken', 'sheep', 'cow', 'bird2', 'DOG', 'doggg',
                          'lizard', 'cat3', 'cat2', 'cat1'])


if __name__ == '__main__':
    unittest.main()
