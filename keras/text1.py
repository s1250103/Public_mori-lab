import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import keras
from keras.datasets import mnist

def plot_mnist(X, y, result_dir): # Xは画像データ, yはラベル(0～9)
    row, col = 10,10 # 10行10列で画像を描画
    plt.figure() # 描画開始
    fig, axes = plt.subplots(
        row,col,figsize=(10,10), # 描画全体のサイズ(単位はインチ)
        gridspec_kw={'wspace':0, 'hspace':0.05}) # 画像間の上下間隔

    # 10種類ごとにまとめて描画
    nclasses = 10
    for targetClass in range(nclasses):
        targetIdx = []
        # クラスclassIDの画像のインデックスリストを取得
        for i in range(len(y)):
            if y[i] == targetClass:
                targetIdx.append(i)

        # 各クラスからランダムに選んだ最初の10個の画像を描画
        np.random.shuffle(targetIdx)
        for i in range(col):
            idx = targetIdx[i]
            img = Image.fromarray(X[idx]) # 実数配列から画像データに変換
            axes[targetClass][i].set_axis_off() # 軸は表示しない
            axes[targetClass][i].imshow(img, cmap=plt.get_cmap('gray'))

    plt.savefig(os.path.join(result_dir, 'MNIST-sample.jpg')) # 画像を保存
    plt.show()

# MNISTデータをロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train) # ラベルデータ（一部）をプリント
print(type(y_train[0])) # ラベルデータ（1番）が整数であることを確認

print("訓練画像データのシェープ：",x_train.shape)
print("テスト画像データのシェープ：",x_test.shape)
print("訓練ラベルデータのシェープ：",y_train.shape)
print("テストラベルデータのシェープ：",y_test.shape)
num_train_data = x_train.shape[0]
num_test_data = x_test.shape[0]

# 各クラス最初の10枚の画像をプロット
plot_mnist(x_train, y_train, 'output')


num_flatten_data = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(num_train_data, num_flatten_data)
x_test = x_test.reshape(num_test_data, num_flatten_data)

num_gray_scale_max = 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= num_gray_scale_max
x_test /= num_gray_scale_max
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# ラベルを整数値からワンホットベクトル値に変換
num_classes = 10 # クラス数(10)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)
print(type(y_train))
print(y_train) #ワンホット・ベクトル変換を確認
