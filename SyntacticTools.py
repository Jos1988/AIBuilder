from typing import List
from nltk.corpus import wordnet


# Add unit test if function is extended beyond use in TestKeyWordToCategoryScrubber.
class AliasLoader:

    def __init__(self, use_synonyms: bool, min_syntactic_distance: float, verbosity: int):
        self.verbosity = verbosity
        self.min_syntactic_distance = min_syntactic_distance
        self.use_synonyms = use_synonyms

    def load_cat_aliases(self, categories: List[str]) -> dict:
        cat_aliases = {}
        for category in categories:
            cat_keywords = self.load_category_keywords(category)
            cat_aliases[category] = cat_keywords

        return cat_aliases

    def load_category_keywords(self, category):
        cat_keywords = [category]
        if False is self.use_synonyms:
            return cat_keywords

        synonyms = self.getSynonyms(category)
        synonyms = self.reformat_synonyms(synonyms)

        if self.verbosity > 0:
            print('found synonyms for "{}": "{}"'.format(category, synonyms))

        cat_keywords = cat_keywords + synonyms

        return cat_keywords

    @staticmethod
    def reformat_synonyms(synonyms: list) -> list:
        formatted = []
        for synonym in synonyms:
            formatted.append(synonym.replace('_', ' '))

        return formatted

    def getSynonyms(self, keyword):
        synonyms = []
        synsets = wordnet.synsets(keyword)
        if len(synsets) is 0:
            return synonyms

        base_synset = synsets[0]
        for synset in synsets:
            if self.check_similarity(base_synset, synset):
                synonyms = synonyms + synset.lemma_names()

        return synonyms

    def check_similarity(self, base_synset, synset):
        similarity = self.get_similarity(base_synset, synset)

        return self.min_syntactic_distance < similarity

    @staticmethod
    def get_similarity(base_synset, synset):
        similarity = base_synset.path_similarity(synset)
        if similarity is None:
            similarity = 0
        return similarity
