
from functions import sigmoid, softmax, numerical_gradient, \
    cross_entropy_error, sigmoid_grad
import numpy as np


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size,
                weight_init_std=0.01):
        """２層のニューラルネットワーク

        Args:
            input_size (int): 入力層のニューロンの数
            hidden_size (int): 隠れ層のニューロンの数
            output_size (int): 出力層のニューロンの数
            weight_init_std (float, optional): 重みの初期値の調整パラメーター。デフォルトは0.01。
        """

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """ニューラルネットワークによる推論

        Args:
            x (numpy.ndarray): ニューラルネットワークへの入力

        Returns:
            numpy.ndarray: ニューラルネットワークの出力
        """
        # パラメーター取り出し
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # ニューラルネットワークの計算（forward）
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """損失関数の値算出

        Args:
            x (numpy.ndarray): ニューラルネットワークへの入力
            t (numpy.ndarray): 正解のラベル

        Returns:
            float: 損失関数の値
        """
        # 推論
        y = self.predict(x)

        # 交差エントロピー誤差の算出
        loss = cross_entropy_error(y, t)

        return loss

    def accuracy(self, x, t):
        """認識精度算出

        Args:
            x (numpy.ndarray): ニューラルネットワークへの入力
            t (numpy.ndarray): 正解のラベル

        Returns:
            float: 認識精度
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / x.shape[0]
        return accuracy

    def numerical_gradient(self, x, t):
        """重みパラメーターに対する勾配の算出

        Args:
            x (numpy.ndarray): ニューラルネットワークへの入力
            t (numpy.ndarray): 正解のラベル

        Returns:
            dictionary: 勾配を格納した辞書
        """
        grads = {}
        grads['W1'] = \
            numerical_gradient(lambda: self.loss(x, t), self.params['W1'])
        grads['b1'] = \
            numerical_gradient(lambda: self.loss(x, t), self.params['b1'])
        grads['W2'] = \
            numerical_gradient(lambda: self.loss(x, t), self.params['W2'])
        grads['b2'] = \
            numerical_gradient(lambda: self.loss(x, t), self.params['b2'])

        return grads

    def gradient(self, x, t):
        """5章で学ぶ関数。誤差逆伝播法の実装
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads