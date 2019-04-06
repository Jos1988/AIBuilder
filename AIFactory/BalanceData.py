import warnings
from abc import abstractmethod, ABC
from typing import List

import pandas as pd

from AIBuilder.Data import DataModel

WEIGHTS_COLUMN = 'weights'


class UnbalancedDataStrategy(ABC):
    OVER_SAMPLING = 'over sampling'
    UNDER_SAMPLING = 'under sampling'
    RE_WEIGH = 're-weigh'

    @abstractmethod
    def balance_data(self, data_model: DataModel, target_column_name: str) -> DataModel:
        pass

    @staticmethod
    @abstractmethod
    def strategy_name() -> str:
        pass

    @staticmethod
    def prepare_data(data_model: DataModel, target_column_name: str):
        stack_one, stack_two = BinaryResampling.separate_by_target_categories(data_model, target_column_name)

        if len(stack_one) == len(stack_two):
            return data_model

        long_stack, short_stack = BinaryResampling.set_long_and_short_stack(stack_one, stack_two)

        return long_stack, short_stack


class BinaryResampling(UnbalancedDataStrategy):

    @staticmethod
    def set_long_and_short_stack(stack_one, stack_two):
        long_stack = stack_two
        short_stack = stack_one

        if len(stack_one) > len(stack_two):
            long_stack = stack_one
            short_stack = stack_two

        return long_stack, short_stack

    @staticmethod
    def separate_by_target_categories(data_model: DataModel, target_column_name):
        df = data_model.get_dataframe()
        categories = df[target_column_name].unique()

        stack_one = df.loc[df[target_column_name] == categories[0]]
        stack_two = df.loc[df[target_column_name] == categories[1]]

        return stack_one, stack_two

    @staticmethod
    def cut_df_to_length(dataframe: pd.DataFrame, length: int) -> pd.DataFrame:
        assert len(dataframe) > length, 'Cannot cut dataframe smaller than required length.'

        return dataframe.head(length)

    @staticmethod
    def validate_result(long_stack, short_stack):
        assert len(short_stack) == len(long_stack)

    @staticmethod
    def merge_stacks(long_stack, short_stack) -> pd.DataFrame:
        scrubbed_df = pd.concat([long_stack, short_stack])
        scrubbed_df = scrubbed_df.sort_index()

        return scrubbed_df


class OverSampling(BinaryResampling):

    def balance_data(self, data_model: DataModel, target_column_name: str) -> DataModel:
        long_stack, short_stack = self.prepare_data(data_model=data_model, target_column_name=target_column_name)

        length_to_have = len(long_stack)

        duplicate_short_stack = short_stack.copy()
        while len(short_stack) < length_to_have:
            short_stack = pd.concat([short_stack, duplicate_short_stack])

        short_stack = self.cut_df_to_length(short_stack, length_to_have)

        self.validate_result(long_stack, short_stack)

        new_df = self.merge_stacks(long_stack, short_stack)
        data_model.set_dataframe(new_df)

        return data_model

    @staticmethod
    def strategy_name() -> str:
        return UnbalancedDataStrategy.OVER_SAMPLING


class UnderSampling(BinaryResampling):

    def balance_data(self, data_model: pd.DataFrame, target_column_name: str) -> pd.DataFrame:
        long_stack, short_stack = self.prepare_data(data_model=data_model, target_column_name=target_column_name)

        long_stack = self.cut_df_to_length(long_stack, len(short_stack))

        self.validate_result(long_stack, short_stack)

        new_df = self.merge_stacks(long_stack, short_stack)
        data_model.set_dataframe(new_df)

        return data_model

    @staticmethod
    def strategy_name() -> str:
        return UnbalancedDataStrategy.UNDER_SAMPLING


class ReWeigh(UnbalancedDataStrategy):

    def balance_data(self, data_model: DataModel, target_column_name: str) -> DataModel:
        weights_by_label = self.map_weights_by_cat_label(data_model, target_column_name)

        df = data_model.get_dataframe()
        weights = self.get_weights_list(df, target_column_name, weights_by_label)

        weights_column_warning = 'note: adding weights column ({}), make sure it is passed to the estimator- and ' \
                                 'data builder!'.format(WEIGHTS_COLUMN)
        warnings.warn(weights_column_warning)

        df[WEIGHTS_COLUMN] = weights
        data_model.set_dataframe(df)

        return data_model

    def map_weights_by_cat_label(self, data_model, target_column_name):
        left_stack, right_stack = self.prepare_data(data_model=data_model, target_column_name=target_column_name)

        average_stack_length = (len(left_stack) + len(right_stack)) / 2
        weight_left = average_stack_length / len(left_stack)
        weight_right = average_stack_length / len(right_stack)

        category_left = self.get_stack_category(left_stack, target_column_name)
        category_right = self.get_stack_category(right_stack, target_column_name)

        weights_by_label = {
            category_left: weight_left,
            category_right: weight_right,
        }

        return weights_by_label

    @staticmethod
    def get_weights_list(df: pd.DataFrame, target_column_name: str, weights_by_label: dict) -> List[float]:
        weights = []
        for item in df[target_column_name].items():
            weights.append(weights_by_label[item[1]])

        return weights

    @staticmethod
    def get_stack_category(left_stack, target_column_name):
        left_stack_categories = left_stack[target_column_name].unique()
        assert 1 is len(left_stack_categories)
        category_left = left_stack_categories[0]

        return category_left

    @staticmethod
    def strategy_name() -> str:
        return UnbalancedDataStrategy.RE_WEIGH


class UnbalancedDataStrategyFactory:
    strategies = [
        OverSampling,
        UnderSampling,
        ReWeigh
    ]  # type: List[UnbalancedDataStrategy]

    @staticmethod
    def get_strategy(required_strategy_name: str):
        for strategy in UnbalancedDataStrategyFactory.strategies:
            if required_strategy_name == strategy.strategy_name():
                return strategy()

        raise RuntimeError('Unbalanced data strategy type ({}) not found.'.format(required_strategy_name))
