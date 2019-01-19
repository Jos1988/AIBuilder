import json
from difflib import SequenceMatcher
from typing import List, Optional


class DuplicationGroupMerger:

    def __init__(self):
        self.sets_to_merge = []

    def merge_duplicate_groups(self, sets_to_merge: List[set]):
        """
        Merge all duplicate sets that hold at least the same item once.
        :return:
        """
        self.sets_to_merge = sets_to_merge

        groups_work_list = self.sets_to_merge.copy()
        groups_work_list_deduplicated = []

        while groups_work_list_deduplicated != groups_work_list:
            groups_work_list_deduplicated = self._do_merge_duplicate_groups(groups_work_list)

        # Update duplicate groups
        return groups_work_list_deduplicated

    @staticmethod
    def _do_merge_duplicate_groups(groups: list) -> list:
        for group in groups:
            # compare duplication groups against each other.
            for compare_group in groups:
                # Skip if were comparing a group with itself.
                if group == compare_group:
                    continue

                # If two duplication groups intersect merge them.
                if bool(group & compare_group):
                    merged_groups = group.union(compare_group)
                    # remove original groups from list where needed.
                    groups.remove(compare_group)
                    if group in groups:
                        groups.remove(group)

                    # Add new group to list
                    groups.append(merged_groups)

        return groups


class AliasListBuilder:
    def __init__(self):
        self.alias_list = {}

    def convert_groups(self, duplicate_groups: List[set]) -> dict:
        for group in duplicate_groups:
            self.update_alias_list_with_group(group)

        return self.alias_list

    def update_alias_list_with_group(self, group):
        selected_group_item = self.select_shortest_item(group)
        for item in group:
            if item is selected_group_item:
                continue
            self.alias_list[item] = selected_group_item

    @staticmethod
    def select_shortest_item(group) -> str:
        shortest_item = list(group)[0]
        shortest_len = len(shortest_item)
        for item in group:
            if len(item) < shortest_len:
                shortest_item = item
        return shortest_item


class StringDeduplicator:

    def __init__(self, subjects: List[str], criterion: float):
        """
        :param subjects: List of unique strings to check for duplicates, (similar strings).
        :param criterion: Threshold if two strings similarity is more than this, they are considered duplicates.
        """
        self.criterion = criterion
        self.compare_list = subjects  # List of all subjects to for comparing to.
        # List of all subjects to compare, items are popped of this list and compared to the compare_list.
        self.work_list = subjects.copy()
        self.deduplicated_list = []  # List of 'unique' subjects.

        # List of groups(set) of duplicates, for example:
        # [['dog' , 'Dog'], ['cat' ,'Cat', 'cats']] # 'Cat', 'cats' and 'cat' are duplicates as are 'dog' and 'Dog'.
        self.duplicate_groups = []
        self.alias_list = {}  # dict for converging values, example: {'cat': 'Cat2', 'cat': 'Cat'}

        self.set_merger = DuplicationGroupMerger()
        self.alias_list_builder = AliasListBuilder()

    def deduplicate(self):
        # get next worklist item until worklist is empty.
        while len(self.work_list) is not 0:
            value = self.work_list.pop()

            # Add item to deduplicated list, duplicates, when found, are removed from the work list
            # and will not appear here.
            self.deduplicated_list.append(value)

            # Check if item is duplicate to any item in the compare list
            # print('checking value: {}, {} values to go.'.format(value, len(self.work_list)))
            self._compare_item(value)

        self.duplicate_groups = self.set_merger.merge_duplicate_groups(self.duplicate_groups)
        self.alias_list = self.alias_list_builder.convert_groups(self.duplicate_groups)

    def _compare_item(self, item: str):
        # Loop over compare list.
        for compare_list_item in self.compare_list:
            # Ignore item if it encounters itself in the compare list.
            if item == compare_list_item:
                # print(item + '==' + compare_list_item)
                continue

            duplicate_group, items_to_compare = self._get_group_and_items_to_compare_to(compare_list_item)
            max_ratio = self._get_best_comparison_score(item, items_to_compare)

            # If item similar to compare list item or its duplicates consider item duplicate of compare list item.
            if max_ratio > self.criterion:
                self._add_to_create_duplicates_group(compare_list_item, duplicate_group, item)
                self._remove_from_worklist(compare_list_item)

    def _remove_from_worklist(self, compare_list_item):
        """ Finally remove item from worklist as its a duplicate. This way we don't do the same work twice. """
        if compare_list_item in self.work_list:
            self.work_list.remove(compare_list_item)

    def _add_to_create_duplicates_group(self, compare_list_item, duplicate_group, item):
        """Add item to duplicates group or create a new one. """
        if duplicate_group is not None:
            self._add_to_duplicate_group(duplicate_group, item)
        else:
            self.duplicate_groups.append({item, compare_list_item})

    def _get_best_comparison_score(self, item, items_to_compare):
        """ Compare item to compare list Items and its duplicates. """
        ratios = []
        for compare_subject in items_to_compare:
            ratios.append(self.compare_values(value1=item, value2=compare_subject))

        return max(ratios)

    def _get_group_and_items_to_compare_to(self, compare_list_item):
        """ If compare list item already has duplicates (similar items), make sure the are also compared to item. """
        items_to_compare = {compare_list_item}
        duplicate_group = self._get_duplicate_group(compare_list_item)
        if duplicate_group is not None:
            items_to_compare = duplicate_group.union(items_to_compare)

        return duplicate_group, items_to_compare

    def _get_duplicate_group(self, item) -> Optional[set]:
        """ Check if item already in group of duplicates and return group if true.

        :param item:
        :return:
        """
        for group in self.duplicate_groups:
            if item in group:
                return group

        return None

    @staticmethod
    def compare_values(value1: str, value2: str) -> float:
        """ Compare two strings to see if the are similar.

        :param value1:
        :param value2:
        :return: float indicating similarity, high is very similar.
        """
        return SequenceMatcher(None, value1.lower(), value2.lower()).ratio()

    def _add_to_duplicate_group(self, duplicate_group: set, item: str) -> None:
        """
        :param duplicate_group: list
        :param item: str
        :return: None
        """
        if item in duplicate_group:
            duplicate_group.remove(item)

        for group in self.duplicate_groups:
            if group == duplicate_group:
                self.duplicate_groups.remove(group)
                duplicate_group.add(item)
                self.duplicate_groups.append(duplicate_group)
                return


class AliasListStorageMover:

    @staticmethod
    def move_to_storage(alias_list: dict, file_name: str):
        json_list = json.dumps(alias_list)
        file = open(file=file_name, mode='x')
        file.write(json_list)
        file.close()

    @staticmethod
    def get_from_storage(file_name: str) -> None:
        file = open(file=file_name, mode='r')
        json_data = file.read()
        file.close()
        dict_data = json.loads(json_data)

        return dict_data


# scrub categorical column using alias dict.

# scrub multiple cat using alias dict.
