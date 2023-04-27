from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
model=Sequential()
model.add(Flatten())
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(10, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mnist=keras.datasets.mnist
(train_data, train_label), (test_data, test_label) = mnist.load_data()
#下載minst檔案
print(type(train_data))
print(train_data.shape)
print(type(train_label))
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)
print(train_label[0])
print(train_data[0])
import matplotlib.pyplot as plt
plt.imshow(train_data[0], cmap='binary')
plt.show()
#輸出第1張由minst檔案集裡下載的圖片
model.fit(train_data,train_label,epochs=20,batch_size=512)
# train_data train_label為訓練資料 每512筆資料分為1個batch 所有的batch訓練20次
score=model.evaluate(test_data, test_label)
print(score[1])
