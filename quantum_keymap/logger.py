import os

import matplotlib.pyplot as plt
import numpy as np


class Logger(object):
    def __init__(self, result_dir, model) -> None:
        self.result_dir = result_dir
        self.model = model
        self.prepare_log_file()

    def prepare_log_file(self):
        log_path = self.result_dir / "log.csv"
        if not os.path.isfile(log_path):
            os.makedirs(self.result_dir, exist_ok=True)
            with open(log_path, "w") as f:
                header_key = [
                    "w_1hot",
                    "w_key_unique",
                    "beta",
                    "gamma",
                    "num_sweeps",
                    "trotter",
                    "num_reads",
                    "energy_avg",
                    "energy_std",
                    "energy_min",
                    "valids",
                    "cost_min",
                    "keys",
                ]
                f.write(",".join(header_key) + "\n")

    def log(self, weight, annealing_params, states):

        energies = np.array([self.model.energy(state, **weight) for state in states])
        valids = np.array([self.model.validate(state) for state in states])
        valid_states = states[valids]
        valid_costs = energies[valids]

        # write statics
        with open(self.result_dir / "log.csv", "a") as f:
            values = [
                weight["w_1hot"],
                weight["w_key_unique"],
                annealing_params["beta"],
                annealing_params["gamma"],
                annealing_params["num_sweeps"],
                annealing_params["trotter"],
                annealing_params["num_reads"],
                np.average(energies),
                np.std(energies),
                np.min(energies),
                sum(valids),
            ]
            values_str = [str(v) for v in values]
            f.write(",".join(values_str) + ",")

        if len(valid_states) == 0:
            print("There are no valid results.")
            with open(self.result_dir / "log.csv", "a") as f:
                f.write(",\n")
            return

        cost_min = valid_costs.min()
        min_state = valid_states[valid_costs.argmin()]
        keymap = self.model.keys_from_state(min_state)

        print("cost_min:", cost_min)
        print("key_map:")
        for row in keymap:
            print(row)

        with open(self.result_dir / "log.csv", "a") as f:
            f.write(str(cost_min) + ',"' + "".join(keymap.flatten()) + '"\n')

        detail_dir = self.result_dir / str(cost_min)
        if not os.path.isdir(detail_dir):
            os.makedirs(detail_dir, exist_ok=True)
        self.save_keymap(keymap, detail_dir)

    def save_keymap(self, keys, dir):
        ly, lx = keys.shape
        fig = plt.figure(figsize=(lx, ly))

        ax = fig.add_subplot()
        ax.set_aspect("equal")

        # plot keyboard grid
        plt.xticks(range(lx + 1))
        plt.yticks(range(ly + 1))
        ax.invert_yaxis()
        ax.grid()

        for y, row in enumerate(keys):
            for x, char in enumerate(row):
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    char,
                    size=20,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        plt.tight_layout()
        # plt.show()
        plt.savefig(dir / "keymap.png")
