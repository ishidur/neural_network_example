import numpy as np
from neuralnet import NeuralNetwork

teach_inputs = [np.matrix([0, 0]), np.matrix(
    [0, 1]), np.matrix([1, 0]), np.matrix([1, 1])]
teach_outputs = [np.matrix([0]), np.matrix(
    [1]), np.matrix([1]), np.matrix([0])]

if __name__ == '__main__':
    learning_time_limit = 10000
    neuralnet = NeuralNetwork([2, 3, 1], 0.1, 0.5)
    for n in range(learning_time_limit):
        for data_no in range(len(teach_inputs)):
            neuralnet.forward_propagate(teach_inputs[data_no])
            neuralnet.back_propagation(teach_outputs[data_no])

    for data_no in range(len(teach_inputs)):
        o = neuralnet.forward_propagate(teach_inputs[data_no])
        print(o, (teach_outputs[data_no] - o)**2 / 2)
