from colorama import Fore, Style
import time

def log_info_time(message, time):
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + message + Style.RESET_ALL, time)


def log_warning(message):
    print(Fore.LIGHTRED_EX + Style.BRIGHT + message + Style.RESET_ALL)


def log_info(message):
    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + message + Style.RESET_ALL)

def time_checker(msg, func,**kwargs):
    start = time.time()
    if len(kwargs) == 0:
        rst = func()
    else :
        rst = func(**kwargs)
    end = time.time()
    log_info_time(msg, end - start)
    return rst
