from datetime import datetime


def datetime_string_to_date(datetime_string: str):
    """

    :param datetime_string:
    :return: date
    """
    # split in to date and time
    split_string = datetime_string.split('T')

    # create date
    return datetime.strptime(split_string[0], '%Y-%m-%d').date()
