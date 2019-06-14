import numpy as np
from neuralnet import NeuralNetwork

teach_inputs = [np.matrix([0, 0]), np.matrix(
    [0, 1]), np.matrix([1, 0]), np.matrix([1, 1])]
teach_outputs = [np.matrix([0]), np.matrix(
    [1]), np.matrix([1]), np.matrix([0])]


def validate(nn, test_inputs, test_outputs):
    err_total = 0.0
    for i in range(len(test_inputs)):
        o = neuralnet.forward_propagate(test_inputs[data_no])
        err = (test_outputs[data_no] - o)**2 / 2
        err_total += err
    return err_total


if __name__ == '__main__':
    learning_time_limit = 5000
    error_boundary = 0.01
    neuralnet = NeuralNetwork([2, 3, 1], 0.1, 1.0)
    for n in range(learning_time_limit):
        for data_no in range(len(teach_inputs)):
            neuralnet.forward_propagate(teach_inputs[data_no])
            neuralnet.back_propagation(teach_outputs[data_no])
            err = validate(neuralnet, teach_inputs, teach_outputs)
            if err <= error_boundary:
                print("finished in %d times of learning" % (n))
                break
        else:
            continue
        break
    for data_no in range(len(teach_inputs)):
        o = neuralnet.forward_propagate(teach_inputs[data_no])
        print(o, (teach_outputs[data_no] - o)**2 / 2)
        print("total")
        print(validate(neuralnet, teach_inputs, teach_outputs))
