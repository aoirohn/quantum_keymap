from collections.abc import Iterable
import re

import numpy as np

import quantum_keymap.config.default as default_conf
from quantum_keymap.util import load_config


class KeymapModel(object):
    def __init__(self, config: dict) -> None:
        self.config = load_config(default_conf)
        self.config.update(config)

        self.key_to_code = {}
        for i, key in enumerate(self.config["KEY_LIST"]):
            if isinstance(key, Iterable):
                for item in key:
                    self.key_to_code[item] = i
            else:
                self.key_to_code[key] = i

        self.code_to_key = {v: k for k, v in self.key_to_code.items()}

        self.N = np.array(self.config["HAND"]).size
        self.J = np.zeros((self.N * self.N, self.N * self.N))
        self.H_1hot, self.const_1hot = self._create_H_1hot()
        self.H_key_unique, self.const_key_unique = self._create_H_key_unique()

        chars = "".join([key.lower() for key in self.key_to_code.keys()])
        self.p = re.compile(f"[{chars}]+")

    def update_weight(self, text):
        J_sub = np.zeros((self.N, self.N, self.N, self.N))
        text = text.lower()
        result = self.p.findall(text)

        position_cost = self.config["POSITION_COST"]
        hand = self.config["HAND"]
        finger = self.config["FINGER"]
        consecutive_hand_cost = self.config["CONSECUTIVE_HAND_COST"]
        consecutive_finger_cost = self.config["CONSECUTIVE_FINGER_COST"]
        consecutive_key_cost = self.config["CONSECUTIVE_KEY_COST"]

        for string in result:
            # position cost
            for char in string:
                code = self.key_to_code[char]
                for key in range(self.N):  # key position
                    J_sub[key][code] += position_cost[key]

            # hand/finger cost
            for pos in range(len(string) - 1):
                char1 = self.key_to_code[string[pos]]
                char2 = self.key_to_code[string[pos + 1]]
                for key1 in range(self.N):
                    for key2 in range(self.N):

                        # add finger cost
                        if hand[key1] == hand[key2]:
                            J_sub[key1, char1][key2, char2] += consecutive_hand_cost

                            # add finger cost
                            if finger[key1] == finger[key2]:
                                if char1 == char2:
                                    J_sub[key1, char1][
                                        key2, char2
                                    ] += consecutive_key_cost
                                else:
                                    J_sub[key1, char1][
                                        key2, char2
                                    ] += consecutive_finger_cost

        J_sub = J_sub.reshape((self.N * self.N, self.N * self.N))
        self.J += J_sub

    def _create_H_1hot(self):
        H = np.zeros((self.N, self.N, self.N, self.N))

        for key in range(self.N):
            for char1 in range(self.N):
                H[key, char1][key, char1] = -1
                for char2 in range(char1 + 1, self.N):
                    H[key, char1][key, char2] = 2
        return H.reshape((self.N * self.N, self.N * self.N)), self.N

    def _create_H_key_unique(self):
        H = np.zeros((self.N, self.N, self.N, self.N))

        for char in range(self.N):
            for key1 in range(self.N):
                H[key1, char][key1, char] = -1
                for key2 in range(key1 + 1, self.N):
                    H[key1, char][key2, char] = 2
        return H.reshape((self.N * self.N, self.N * self.N)), self.N

    def H(self, w_1hot, w_key_unique):
        H = self.J + w_1hot * self.H_1hot + w_key_unique * self.H_key_unique
        const = w_1hot * self.const_1hot + w_key_unique * self.const_key_unique
        return H, const

    def energy(self, state, w_1hot, w_key_unique):
        H, const = self.H(w_1hot, w_key_unique)
        return state @ H @ state + const

    def cost(self, state):
        return state @ self.J @ state

    def _energy_1hot(self, state, weight):
        return weight * (state @ self.H_1hot @ state + self.const_1hot)

    def _energy_key_unique(self, state, weight):
        return weight * (state @ self.H_key_unique @ state + self.const_key_unique)

    def _energy_key_unique(self, qbits, weight):
        return qbits @ (weight * self.H_key_unique) @ qbits

    def qubo(self, w_1hot, w_key_unique):
        H, _ = self.H(w_1hot, w_key_unique)
        H = np.triu(H, k=1) + np.triu(H.T)

        qubo = {}
        size = len(H)
        for i in range(size):
            for j in range(i, size):
                if H[i][j] != 0:
                    qubo[(i, j)] = H[i][j]
        return qubo

    def keys_from_state(self, state):
        state_2d = state.reshape((-1, self.N))
        code = np.dot(state_2d, np.array(range(self.N))).astype(int)
        keys = np.array([self.code_to_key[c] for c in code])
        return keys.reshape(np.array(self.config["HAND"]).shape)
