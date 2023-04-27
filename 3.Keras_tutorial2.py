from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
 
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=784))
# 512個神經元 因為是中間層所以用relu 輸入784個維度
model.add(Dropout(0.2))
# 減少20%的神經元
model.add(Dense(10, activation='softmax'))
# 輸出層10個神經元，兩個以上神經元用softmax
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# optimizer優化器 loss是損失函數 metrics是準確率
import numpy as np
data = np.random.random((1000, 784)) 
#1000個list 裡面有784個浮點數
labels = np.random.randint(10, size=(1000, 10))
#產生0到9的隨機數字 並放到2維陣列裡
model.fit(data, labels, epochs=10, batch_size=32)
# data labels為訓練資料 每32筆資料分為1個batch 所有的batch訓練10次
score=model.evaluate(data, labels)
# 正確率輸出
print(score[1])
# [loss , accuracy]
