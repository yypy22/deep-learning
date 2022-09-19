
import numpy as np
import matplotlib.pylab as plt
import os
import sys
from two_layer_net import TwoLayerNet
sys.path.append(os.pardir)  # パスに親ディレクトリ追加
from data import load_mnist


# MNISTの訓練データとテストデータ読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# ハイパーパラメーター設定
iters_num = 10000       # 更新回数
batch_size = 100        # バッチサイズ
learning_rate = 0.1     # 学習率

# 結果の記録リスト
train_loss_list = []    # 損失関数の値の推移
train_acc_list = []     # 訓練データに対する認識精度
test_acc_list = []      # テストデータに対する認識精度

train_size = x_train.shape[0]  # 訓練データのサイズ
iter_per_epoch = max(train_size / batch_size, 1)    # 1エポック当たりの繰り返し数

# 2層のニューラルワーク生成
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 学習開始
for i in range(iters_num):

    # ミニバッチ生成
    batch_mask = np.random.choice(train_size, batch_size, replace=False)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    # grad = network.numerical_gradient(x_batch, t_batch)  遅いので誤差逆伝搬法で……
    grad = network.gradient(x_batch, t_batch)

    # 重みパラメーター更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 損失関数の値算出
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポックごとに認識精度算出
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # 経過表示
        print(f"[更新数]{i: >4} [損失関数の値]{loss:.4f} "
                f"[訓練データの認識精度]{train_acc:.4f} [テストデータの認識精度]{test_acc:.4f}")

# 損失関数の値の推移を描画
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='loss')
plt.xlabel("iteration")
plt.ylabel("loss")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()

# 訓練データとテストデータの認識精度の推移を描画
x2 = np.arange(len(train_acc_list))
plt.plot(x2, train_acc_list, label='train acc')
plt.plot(x2, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.xlim(left=0)
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()