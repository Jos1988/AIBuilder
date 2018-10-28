from currency_converter import CurrencyConverter
from datetime import date


# todo: deprecated
class Converter:
    _default_currency = 'USD'
    _converter = CurrencyConverter(fallback_on_missing_rate=True)

    def __init__(self, default_currency: str = None):
        if default_currency is not None:
            self._default_currency = default_currency

    def convert_to_default(self, amount: int, from_currency: str, exchange_rate_date: date = None):

        if from_currency is self._default_currency:
            return amount

        return self._converter.convert(amount, from_currency, new_currency=self._default_currency, date=exchange_rate_date)
