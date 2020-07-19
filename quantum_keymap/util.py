from types import ModuleType


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
