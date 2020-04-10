#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 3-6-predicting-house-prices.py
@time: 2020/4/9 16:51
@project: deep-learning-with-python-notebooks
@desc: 3.6 预测房价: 回归问题
"""

from tensorflow.keras import models, layers
from tensorflow.keras.datasets import boston_housing

# 加载数据
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据预处理：归一化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# 模型构建
def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# Get a fresh, compiled model.
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=1)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mae_score)
