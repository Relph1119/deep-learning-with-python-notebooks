#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 3-4-classifying-movie-reviews.py
@time: 2020/4/9 14:48
@project: deep-learning-with-python-notebooks
@desc: 3.4 电影评论分类：二分类问题
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers, losses, metrics, optimizers
from tensorflow.keras.datasets import imdb

# 加载电影评论数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# 数据集进行向量化
def vectorize_sequences(sequences, dimension=10000):
    """
    将整数序列编码为二进制矩阵
    :param sequences:
    :param dimension:
    :return:
    """
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 模型构建
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 模型装配
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

# 绘制图形
loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

epochs = range(1, len(loss_value) + 1)

plt.subplots_adjust(wspace=0.5)
plt.subplot(1, 2, 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss_value, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss_value, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

results = model.evaluate(x_test, y_test)
print(results)

print(model.predict(x_test))
