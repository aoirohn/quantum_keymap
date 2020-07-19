from types import ModuleType
from itertools import chain


def load_config(module: ModuleType):
    dic = {}

    for key in [
        "HAND",
        "FINGER",
        "POSITION_COST",
        "CONSECUTIVE_HAND_COST",
        "CONSECUTIVE_FINGER_COST",
        "CONSECUTIVE_KEY_COST",
        "KEY_LIST",
    ]:
        if key in vars(module):
            dic[key] = vars(module)[key]

    return dic


def list_concat(lists):
    return list(chain.from_iterable(lists))

