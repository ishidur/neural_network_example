############################################################
# Ref: https://qiita.com/haltaro/items/7639208417a751ad9bab
############################################################

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def differential_sigmoid(y):
    return np.multiply(y, (1.0 - y))


class NeuralNetwork:
    weights = []
    biases = []
    outputs = []
    learning_rate = 1.0

    def __init__(self, structure, init_val):
        for i in range(len(structure) - 1):
            self.weights.append(np.matrix(
                np.random.uniform(-init_val, init_val, (structure[i], structure[i + 1]))))
            self.biases.append(np.matrix(np.random.uniform(-init_val,
                                                           init_val, (structure[i + 1]))))

    def forward_propagate(self, inputs):
        self.outputs = []
        self.outputs.append(inputs)
        for i in range(len(self.biases)):
            layer_inputs = np.matmul(
                self.outputs[i], self.weights[i]) + self.biases[i]
            self.outputs.append(sigmoid(layer_inputs))
        return self.outputs[-1]

    def __calc_delta(self, prev_delta, layer_index):
        return np.multiply(np.matmul(prev_delta, self.weights[layer_index + 1].T), differential_sigmoid(self.outputs[layer_index + 1]))

    def back_propagation(self, teach_data):
        layer_index = 1
        diff_err = (neuralnet.outputs[-layer_index] - teach_data)
        delta = np.matmul(diff_err, differential_sigmoid(
            self.outputs[-layer_index]))
        self.weights[-layer_index] -= self.learning_rate * \
            np.matmul(self.outputs[-(layer_index + 1)].T, delta)
        self.biases[-layer_index] -= self.learning_rate * delta
        for i in range(len(self.biases) - 2, -1, -1):
            delta = self.__calc_delta(delta, i)
            self.weights[i] -= self.learning_rate * \
                np.matmul(self.outputs[i].T, delta)
            self.biases[i] -= self.learning_rate * self.learning_rate * delta


if __name__ == '__main__':
    # np.arrayではなくnp.matrixを使う．arrayだと，誤差逆伝播法の出力層の重みの更新量の計算でdeltaがscalarになって詰むので気をつけましょう.
    teach_inputs = [np.matrix([0, 0]), np.matrix(
        [0, 1]), np.matrix([1, 0]), np.matrix([1, 1])]
    teach_outputs = [np.matrix([0]), np.matrix(
        [1]), np.matrix([1]), np.matrix([0])]
    neuralnet = NeuralNetwork([2, 3, 1], 0.1)
    for n in range(5000):
        for data_no in range(len(teach_inputs)):
            neuralnet.forward_propagate(teach_inputs[data_no])
            neuralnet.back_propagation(teach_outputs[data_no])

    for data_no in range(len(teach_inputs)):
        o = neuralnet.forward_propagate(teach_inputs[data_no])
        print(o, (teach_outputs[data_no] - o)**2 / 2)
    # print(neuralnet.weights)
