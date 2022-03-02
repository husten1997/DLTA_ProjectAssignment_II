from math import floor, ceil


def prompt(message: str, formatter = 'plain', promt_length = 80):
    """
    Helper function which prints a formatted console messages.

    :param message: str message
    :param formatter: desired formatting
    :param promt_length: desired message length, default is 80 chars
    :return:
    """
    print(messageFormatter(message, promt_length))


def messageFormatter(message: str, prompt_length, fill_char ="-"):
    message = " " + message.strip() + " "
    fill_length = (prompt_length - len(message)) / 2
    prefix = str(fill_char * floor(fill_length))
    postfix = str(fill_char * ceil(fill_length))
    return prefix + message + postfix + "\n"
