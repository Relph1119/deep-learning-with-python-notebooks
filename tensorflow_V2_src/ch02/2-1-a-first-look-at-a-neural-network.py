#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 2-1-a-first-look-at-a-neural-network.py
@time: 2020/4/9 11:27
@project: deep-learning-with-python-notebooks
@desc: 2.1 初识神经网络
"""

import tensorflow.keras as keras
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

print(keras.__version__)

# 加载MNIST数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 构建网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 自动装配
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 处理训练数据，将数据打平
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 进行独热编码转换
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 模型训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 模型评估，得到测试集的ACC
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)
