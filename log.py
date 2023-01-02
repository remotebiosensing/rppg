from colorama import Fore, Style
from params import params
import time

def log_info_time(message, time):
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + message + Style.RESET_ALL, time)


def log_warning(message):
    print(Fore.LIGHTRED_EX + Style.BRIGHT + message + Style.RESET_ALL)


def log_info(message):
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + message + Style.RESET_ALL)

def time_checker(msg, func,**kwargs):
    if params.__TIME__:
        start = time.time()
    rst = func(kwargs)
    if params.__TIME__:
        end = time.time()
        log_info_time(msg, end - start)
    return rst
