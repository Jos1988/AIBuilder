from collections import Counter
from typing import List, Union
from nltk.corpus import wordnet

from AIBuilder.AIFactory.Specifications import DataTypeSpecification


# Add unit test if function is extended beyond use in TestKeyWordToCategoryScrubber.
class SynonymLoader:
    """ Loads synonyms for a word or a list of words. """

    def __init__(self, min_syntactic_distance: float, verbosity: int = 0):
        """

        Args:
            min_syntactic_distance: minimum difference between two words for them to be considerd synonyms by loader.
            verbosity:
        """
        self.verbosity = verbosity
        self.min_syntactic_distance = DataTypeSpecification('min_syntactic_distance', min_syntactic_distance, float)

    def load_synonyms_by_words(self, words: List[str]) -> dict:
        """ Pass a list of words to this method and get a dict of synoyms back

        example:
            loader.load_synonyms_by_words(['job', 'in'])
            will return:
            {'job': ['job', 'occupation', 'business', 'line of work', 'line'], 'in': ['inch', 'in']}

        Args:
            words: List of Words

        Returns: Dict with words as keys and lists of synonyms as values.
        """
        word_synonym_map = {}
        for word in words:
            synonyms = self.load_synonyms(word)
            word_synonym_map[word] = synonyms

        return word_synonym_map

    def load_synonyms(self, word: str) -> List[str]:
        """ Returns List of synonyms to word

        Args:
            word:

        Note:
            - Formats the synonyms.
            - Only returns synonyms with less than te maximum syntactic distance.

        Returns: List of words that are synonyms with respect to the word passed to the method.

        """
        synonyms = self.get_synonyms(word)
        if word not in synonyms:
            synonyms.append(word)
        synonyms = self.reformat_synonyms(synonyms)

        if self.verbosity > 0:
            print(f'found synonyms for "{word}": "{synonyms}"')

        return synonyms

    @staticmethod
    def reformat_synonyms(synonyms: List[str]) -> List[str]:
        """ Removes underscores returned by nltk. """
        formatted = []
        for synonym in synonyms:
            formatted.append(synonym.replace('_', ' '))

        return formatted

    def get_synonyms(self, keyword) -> List[str]:
        """ Gets raw synonyms. """
        synonyms = []
        synsets = wordnet.synsets(keyword)
        if len(synsets) is 0:
            return synonyms

        base_synset = synsets[0]
        for synset in synsets:
            if self.check_similarity(base_synset, synset):
                synonyms = synonyms + synset.lemma_names()

        return synonyms

    def check_similarity(self, base_synset, synset) -> bool:
        """ Checks if similarity is large enough. """
        similarity = self.get_similarity(base_synset, synset)

        return self.min_syntactic_distance() < similarity

    @staticmethod
    def get_similarity(base_synset, synset) -> float:
        """Get similarity between words."""
        similarity = base_synset.path_similarity(synset)
        if similarity is None:
            similarity = 0
        return similarity


class StringCategorizer:
    """ Categorizes a string by finding keywords.

    With multiple set to False the category mapped to the most frequent occurring categories will be assigned, if two
    categories occur in equal measure the first is chosen.

    example 1:
        string = 'Would you like some ham, spam and eggs, sir?'
        map: {'awful': ['spam'],'tasty': ['ham', 'eggs']}
        categorizer = StringCategorizer(map, 'unknown', multiple=False)
        result = categorizer.categorize(string)
        returns: 'tasty' as keywords mapped to the tasty category are most common in the given string.

    With multiple set to True the all found categories are returned in a set.

    example 2:
        string = 'Would you like some ham, spam and eggs, sir?'
        map: {'awful': ['spam'],'tasty': ['ham', 'eggs']}
        categorizer = StringCategorizer(map, 'unknown', multiple=False)
        result = categorizer.categorize(string)
        returns: ('awful', 'tasty').

    Note:
        If no keywords are found 'unknown' category is returned, either as a string or in a set.

    """

    def __init__(self, category_keywords_map: dict, unknown_category: str = 'unknown', multiple: bool = False,
                 verbosity: int = 0):
        """

        Args:
            category_keywords_map: dict mappin keywords to categories
                example: {cat1: (keys1, ...key_n). cat2: (keys1, ...key_n), ... cat_n: (keys1, ...key_n)}
            unknown_category: string that indicates no category could be determined.
            multiple: Whether to return multiple categories or the most occuring keywords.
            verbosity:
        """
        self.category_keywords_map = category_keywords_map
        self.unknown_category = unknown_category
        self.multiple = multiple
        self.verbosity = verbosity

    def categorize(self, string: str) -> Union[str, set]:
        cats_found = self.find_categories(string)

        if len(cats_found) is 0:
            if self.multiple:
                return {self.unknown_category}

            return self.unknown_category

        if self.multiple:
            return set(cats_found)

        # Count most frequent category found in string.
        cat_found = Counter(cats_found).most_common(1)[0][0]
        return cat_found

    def find_categories(self, string: str) -> List[str]:
        """ Finds all categories in string using category to keyword map.

        Args:
            string: String to search.

        Returns: List of categories found, empty if none where found.

        """
        categories_found = []
        for category, key_words in self.category_keywords_map.items():
            for key_word in key_words:
                if key_word.lower() + ' ' in string.lower() + ' ':
                    self.display_association(category, key_word)
                    categories_found.append(category)

        return categories_found

    def display_association(self, category: str, key_word: str) -> None:
        if self.verbosity < 2:
            return

        print(f"Found {category} using keyword {key_word}.")
