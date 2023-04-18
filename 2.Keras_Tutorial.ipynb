from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
data = np.random.random((1000, 10)) #隨機產生1000筆資料，每筆資料包含十個float數字
labels = np.random.randint(2, size=(1000, 1)) #隨機產生1000筆資料，每筆資料上限為2(意思就是0 或 1)
print(np.shape(data))
print(data[0])
print(np.shape(labels))
print(labels[0])

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=10))
# 32個神經元 因為是中間層所以用relu 輸入10個維度
model.add(Dense(1, activation='sigmoid'))
# 最後一層是輸出層，且只有一個神經元使用sigmoid 兩個以上神經元用softmax
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# optimizer優化方法 loss選擇損失函數 metrics成效衡量方式
model.fit(data, labels, epochs=10, batch_size=32)
# 訓練1000筆資料 每32筆資料分為1個batch 所有的batch訓練10次
score=model.evaluate(data, labels)
# 正確率輸出
print(score[1])
# [loss , accuracy]
