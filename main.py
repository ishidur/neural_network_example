############################################################
# Ref: https://qiita.com/haltaro/items/7639208417a751ad9bab
############################################################

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def differential_sigmoid(y):
    return y * (1.0 - y)


class NeuralNetwork:
    weights = []
    biases = []
    outputs = []
    learning_rate = 1.0

    def __init__(self, structure, init_val):
        for i in range(len(structure) - 1):
            self.weights.append(
                np.random.uniform(-init_val, init_val, (structure[i], structure[i + 1])))
            self.biases.append(np.random.uniform(-init_val,
                                                 init_val, (structure[i + 1])))

    def forward_propagate(self, inputs):
        self.outputs = []
        self.outputs.append(inputs)
        for i in range(len(self.biases)):
            layer_inputs = np.dot(
                self.outputs[i], self.weights[i]) + self.biases[i]
            self.outputs.append(sigmoid(layer_inputs))

    def calc_delta(self, prev_delta, layer_index):
        return np.dot(prev_delta, self.weights[layer_index].T) * differential_sigmoid(self.outputs[layer_index + 1])

    def back_propagation(self, teach_data):
        layer_index = 1
        diff_err = (teach_outputs[teach_index] -
                    neuralnet.outputs[-layer_index])
        delta = np.dot(diff_err, differential_sigmoid(
            self.outputs[-layer_index]))
        print(np.dot(self.outputs[-(layer_index + 1)].T, delta).shape)
        print(self.weights[-layer_index].shape)
        d_weight = self.learning_rate * \
            np.dot(self.outputs[-(layer_index + 1)].T, delta)
        print(self.weights[-layer_index])
        self.weights[-layer_index] = self.weights[-layer_index] - d_weight
        print(self.weights[-layer_index])
        self.biases[-layer_index] - self.learning_rate * delta


if __name__ == '__main__':
    # np.arrayではなくnp.matrixを使う．arrayだと，誤差逆伝播法の出力層の重みの更新量の計算でdeltaがscalarになって詰むので気をつけましょう.
    teach_inputs = [np.array([0, 0]), np.array(
        [0, 1]), np.array([1, 0]), np.array([1, 1])]
    teach_outputs = [np.array([0]), np.array(
        [1]), np.array([1]), np.array([1])]
    # teach_inputs = [np.matrix([0, 0]), np.matrix(
    #     [0, 1]), np.matrix([1, 0]), np.matrix([1, 1])]
    # teach_outputs = [np.matrix([0]), np.matrix(
    #     [1]), np.matrix([1]), np.matrix([1])]
    neuralnet = NeuralNetwork([2, 3, 1], 0.1)
    teach_index = 0
    neuralnet.forward_propagate(teach_inputs[teach_index])
    neuralnet.back_propagation(teach_outputs[teach_index])
    # print(sigmoid(teach_inputs[3]))
