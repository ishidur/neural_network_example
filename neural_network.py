############################################################
# Ref: https://qiita.com/haltaro/items/7639208417a751ad9bab
############################################################
import numpy as np
from typing import List
from activation_funcs import sigmoid as activation_func
from activation_funcs import differential_sigmoid as differential_activation_func


class NeuralNetwork:
    weights = []
    biases = []
    outputs = []
    learning_rate = 1.0

    def __init__(self, structure: List[int], init_val: float, learning_rate: float):
        """
        初期化

        Parameters
        ----------
        structure : List[int]
            ニューラルネットワークの構造．
        init_val : float
            重み，しきい値の初期値の取りうる値の範囲．[-init_val, init_val]
        learning_rate : float
            重み，しきい値の学習係数
        """
        self.learning_rate = learning_rate
        for i in range(len(structure) - 1):
            self.weights.append(
                np.matrix(
                    np.random.uniform(
                        -init_val, init_val, (structure[i], structure[i + 1])
                    )
                )
            )
            self.biases.append(
                np.matrix(np.random.uniform(-init_val, init_val, (structure[i + 1])))
            )

    def forward_propagate(self, inputs):
        """
        順伝播計算．このときに各層での出力値を保存するため，学習を行う場合には必ず次の順伝播計算の前にback_propagationを呼ぶこと

        Parameters
        ----------
        inputs : numpy.matrix
            入力値
        """
        self.outputs = []
        self.outputs.append(inputs)
        for i in range(len(self.biases)):
            layer_inputs = np.matmul(self.outputs[i], self.weights[i]) + self.biases[i]
            self.outputs.append(activation_func(layer_inputs))
        return self.outputs[-1]

    def __calc_delta(self, prev_delta, layer_index):
        """
        前のデルタから次のデルタを計算する

        Parameters
        ----------
        prev_delta : numpy.matrix
            一つ前（出力層に近い）の層のデルタ
        layer_index : int
            計算を行っている層の位置
        """
        return np.multiply(
            np.matmul(prev_delta, self.weights[layer_index + 1].T),
            differential_activation_func(self.outputs[layer_index + 1]),
        )

    def back_propagation(self, teach_data):
        """
        誤差逆伝播．教師信号をもとにweights，biasesを更新する．
        self.outputsを利用して更新を行っているため，呼び出すタイミングに注意が必要．

        Parameters
        ----------
        teach_data : numpy.matrix
            教師信号
        """
        layer_index = 1
        diff_err = self.outputs[-layer_index] - teach_data
        delta = np.matmul(
            diff_err, differential_activation_func(self.outputs[-layer_index])
        )
        self.weights[-layer_index] -= self.learning_rate * np.matmul(
            self.outputs[-(layer_index + 1)].T, delta
        )
        self.biases[-layer_index] -= self.learning_rate * delta
        for i in range(len(self.biases) - 2, -1, -1):
            delta = self.__calc_delta(delta, i)
            self.weights[i] -= self.learning_rate * np.matmul(self.outputs[i].T, delta)
            self.biases[i] -= self.learning_rate * self.learning_rate * delta
