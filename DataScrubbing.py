from abc import ABC, abstractmethod
from AIBuilder.Data import DataModel, MetaData
from currency_converter import CurrencyConverter
from datetime import datetime


class Scrubber(ABC):

    @property
    @abstractmethod
    def scrubber_config_list(self):
        pass

    @abstractmethod
    def validate(self, data_model: DataModel):
        pass

    def validate_metadata(self, meta_data: MetaData):
        self._validate_column_config_dict(meta_data)

    @abstractmethod
    def update_metadata(self, meta_data: MetaData):
        pass

    @abstractmethod
    def scrub(self, data_model: DataModel) -> DataModel:
        pass

    def _validate_column_config_dict(self, meta_data: MetaData):
        for column, data_type in self.scrubber_config_list.items():
            if data_type is not meta_data.get_column_type(column):
                raise RuntimeError('scrubber {} validation: column {} should be of data type {}, type {} found'
                                   .format(self.__class__, column, data_type, meta_data.get_column_type(column)))


class MissingDataScrubber(Scrubber):

    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, missing_category_name: str):
        self._missing_category_name = missing_category_name

    def validate(self, data_model: DataModel):
        pass

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        categorical_columns = data_model.metadata.categorical_columns

        return self._scrub_categorical_data(data_model, categorical_columns)

    def _scrub_categorical_data(self, data_model: DataModel, categorical_columns: list) -> DataModel:
        data_model.validate_columns(categorical_columns)
        df = data_model.get_dataframe()
        df[categorical_columns] = df[categorical_columns].fillna(self._missing_category_name)
        data_model.set_dataframe(df)

        return data_model


class StringToDateScrubber(Scrubber):

    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, date_columns: dict, ):
        self.date_columns = date_columns

    def validate(self, data_model: DataModel):
        for date_column, format in self.date_columns.items():
            data_model.validate_columns([date_column])

    def update_metadata(self, meta_data: MetaData):
        pass

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()

        for date_column, format in self.date_columns.items():
            split_format = format.split('T')
            date_format = split_format[0]

            def convert(value):
                return datetime.strptime(
                    value[date_column].split('T')[0],
                    date_format
                )

            df[date_column] = df.apply(convert, axis=1)


class AverageColumnScrubber(Scrubber):

    @property
    def scrubber_config_list(self):
        required_column_config = {self.new_average_column: None}

        for input_column in self._input_columns:
            required_column_config[input_column] = MetaData.NUMERICAL_DATA_TYPE

        return required_column_config

    def __init__(self, input_columns: tuple, output_column: str):
        self._input_columns = input_columns
        self.new_average_column = output_column

    def validate(self, data_model: DataModel):
        data_model.validate_columns(self._input_columns)

    def update_metadata(self, meta_data: MetaData):
        meta_data.define_numerical_columns([self.new_average_column])

        return meta_data

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()
        df[self.new_average_column] = 0

        for input_column in self._input_columns:
            df[self.new_average_column] = df[self.new_average_column] + df[input_column]

        df[self.new_average_column] = df[self.new_average_column] / len(self._input_columns)
        data_model.set_dataframe(df)

        return data_model


class ConvertCurrencyScrubber(Scrubber):

    @property
    def scrubber_config_list(self):
        required_column_config = {self.from_currency_column: MetaData.CATEGORICAL_DATA_TYPE,
                                  self.value_column: MetaData.NUMERICAL_DATA_TYPE}

        if self.exchange_rate_date_column:
            required_column_config[self.exchange_rate_date_column] = MetaData.UNKNOWN_DATA_TYPE

        return required_column_config

    def __init__(self, value_column: str, new_value_column: str, from_currency_column: str, to_currency: str,
                 original_date_column: str = None):

        self.value_column = value_column
        self.new_value_column = new_value_column
        self.from_currency_column = from_currency_column
        self.to_currency = to_currency
        self.exchange_rate_date_column = original_date_column
        self.converter = CurrencyConverter(fallback_on_missing_rate=True, fallback_on_wrong_date=True)

    def validate(self, data_model: DataModel):
        required_columns = [self.value_column, self.from_currency_column]

        if self.exchange_rate_date_column is not None:
            required_columns.append(self.exchange_rate_date_column)

        data_model.validate_columns(required_columns)

    def update_metadata(self, meta_data: MetaData):
        if meta_data.get_column_type(self.new_value_column) is None:
            meta_data.define_numerical_columns([self.new_value_column])

    def scrub(self, data_model: DataModel) -> DataModel:
        df = data_model.get_dataframe()

        def convert_func(value):
            date_argument = None
            if self.exchange_rate_date_column is not None:
                date_argument = value[self.exchange_rate_date_column]

            return self.converter.convert(value[self.value_column],
                                          value[self.from_currency_column],
                                          new_currency=self.to_currency,
                                          date=date_argument)

        df[self.new_value_column] = df.apply(convert_func, axis=1)
        data_model.set_dataframe(df)

        return data_model


class AndScrubber(Scrubber):
    @property
    def scrubber_config_list(self):
        return {}

    def __init__(self, *scrubbers: Scrubber):
        self.scrubber_list = []
        for scrubber in scrubbers:
            self.add_scrubber(scrubber)

    def add_scrubber(self, scrubber: Scrubber):
        self.scrubber_list.append(scrubber)

    def validate(self, data_model: DataModel):
        pass

    def validate_metadata(self, meta_data: MetaData):
        for scrubber in self.scrubber_list:
            scrubber.validate_metadata(meta_data)
            scrubber.update_metadata(meta_data)

    def update_metadata(self, meta_data: MetaData):
        for scrubber in self.scrubber_list:
            scrubber.update_metadata(meta_data)

    def scrub(self, data_model: DataModel) -> DataModel:
        for scrubber in self.scrubber_list:
            scrubber.validate(data_model)
            scrubber.scrub(data_model)

        return data_model
