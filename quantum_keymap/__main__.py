from pathlib import Path

import openjij as oj

import quantum_keymap.config.default as default_conf
from quantum_keymap.logger import Logger
from quantum_keymap.model import KeymapModel
from quantum_keymap.util import load_config


def main():
    conf = load_config(default_conf)
    model = KeymapModel(conf)

    text_path = Path("text/alice.txt")

    weight = {
        "w_1hot": 3000,
        "w_key_unique": 3000,
    }
    annealing_params = {
        "beta": 0.01,
        "gamma": 1000,
        "num_sweeps": 10000,
        "num_reads": 100,
    }

    logger = Logger(Path("result"), model)

    print("building model")
    with open(text_path, "r") as f:
        text = f.readline()
        while text:
            model.update_weight(text)
            text = f.readline()

    qubo = model.qubo(**weight)
    print("annealing")
    sampler = oj.SQASampler(**annealing_params)
    result = sampler.sample_qubo(qubo)

    states = result.states
    logger.log(weight, annealing_params, states)


if __name__ == "__main__":
    main()
