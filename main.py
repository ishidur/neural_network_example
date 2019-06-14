import numpy as np
from neural_network import NeuralNetwork

teach_inputs = [
    np.matrix([0, 0]),
    np.matrix([0, 1]),
    np.matrix([1, 0]),
    np.matrix([1, 1]),
]
teach_outputs = [np.matrix([0]), np.matrix([1]), np.matrix([1]), np.matrix([0])]


def validate(nn, test_inputs, test_outputs):
    """
    検証用データから平均二乗誤差和を計算する

    Parameters
    ----------
    nn : NeuralNetwork
        学習中のニューラルネットワーク
    test_inputs : list
        検証用の入力データ群
    test_outputs : list
        検証用の正解データ群
    """
    err_total = 0.0
    for i in range(len(test_inputs)):
        o = neuralnet.forward_propagate(test_inputs[data_no])
        err = (test_outputs[data_no] - o) ** 2 / 2
        err_total += err
    return err_total


if __name__ == "__main__":
    learning_time_limit = 10000  # 学習回数の上限(打ち切り条件)
    error_boundary = 0.01  # 誤差値によるの終了判定基準
    neuralnet = NeuralNetwork([2, 3, 1], 0.1, 1.0)
    data_indexs = np.arange(len(teach_inputs))
    for n in range(learning_time_limit):
        np.random.shuffle(data_indexs)
        for data_no in data_indexs:
            # 順伝播計算
            neuralnet.forward_propagate(teach_inputs[data_no])
            # 逆伝播学習
            neuralnet.back_propagation(teach_outputs[data_no])
        # 検証
        err = validate(neuralnet, teach_inputs, teach_outputs)
        if err <= error_boundary:
            print("finished in %d times of learning" % (n))
            break
    for data_no in range(len(teach_inputs)):
        o = neuralnet.forward_propagate(teach_inputs[data_no])
        print(o, (teach_outputs[data_no] - o) ** 2 / 2)
        print("total")
        print(validate(neuralnet, teach_inputs, teach_outputs))
