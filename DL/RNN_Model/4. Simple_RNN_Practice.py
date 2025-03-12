# 📌 문제: 간단한 RNN 모델 구현하기
# 다음 요구사항을 만족하는 RNN 모델을 직접 만들어 보세요.

# 🔹 문제 설명
# 주어진 시계열 데이터에서 이전 3개 값을 이용해 다음 값을 예측하는 간단한 RNN 모델을 구현하세요.

# 🔹 요구사항
# 입력 데이터:

# 리스트 data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 를 사용
# window_size=3 를 사용하여 학습 데이터를 생성할 것
# 모델 설계

# TensorFlow의 Sequential 모델을 사용
# SimpleRNN 층을 포함할 것
# 적절한 활성화 함수와 출력층을 설정할 것
# 모델 학습

# epochs=100 으로 설정
# batch_size=1 로 설정
# 새로운 입력 [70, 80, 90] 을 주었을 때 예측값을 출력

# 🔹 추가 조건
# train_test_split()을 사용하여 데이터의 80%는 학습, 20%는 테스트에 사용할 것
# MinMaxScaler를 이용해 데이터를 0~1 범위로 정규화할 것
# 🔹 목표
# 코드를 작성하여 모델을 훈련하고 예측값을 출력하세요.
# (정답이 맞는지 확인하려면 나중에 요청하세요! 😉)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

train_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
window_size = 3

def make_data(sequence_data, cut_size):
    x_total_list = []
    y_total_list = []
    for i in range(len(sequence_data)-cut_size+1):
        x_list = []
        y_list = []
        for j in range(cut_size):
            if j < 2:
                x_list.append(sequence_data[i+j])
            else:
                y_list.append(sequence_data[i+j])

        x_total_list.append(x_list)
        y_total_list.append(y_list)
    
    x_total_list = np.asarray(x_total_list) # [[10, 20], [20, 30], [30, 40], ...]
    y_total_list = np.asarray(y_total_list) # [[30], [40], [50], ...]

    return x_total_list, y_total_list


x, y = make_data(train_list, window_size)

# print(x, y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

mms_for_x = MinMaxScaler()
mms_for_y = MinMaxScaler()

print(X_train.shape)

scaled_X_train = mms_for_x.fit_transform(X_train)
scaled_X_test = mms_for_x.transform(X_test)
scaled_y_train = mms_for_y.fit_transform(y_train)
scaled_y_test = mms_for_y.transform(y_test)

final_X_train = scaled_X_train.reshape(scaled_X_train.shape[0], scaled_X_train.shape[1], 1)
final_X_test = scaled_X_test.reshape(scaled_X_test.shape[0], scaled_X_test.shape[1], 1)

model = Sequential([
        SimpleRNN(10, activation="tanh", return_sequences=True, input_shape=(2, 1)),
        SimpleRNN(30, activation="tanh", return_sequences=True),
        SimpleRNN(15, activation="tanh"),
        Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(final_X_train, scaled_y_train, epochs=700, validation_split=0.2)

test_input = np.asarray([110, 120])
test_input = test_input.reshape(1, -1)

print(test_input.shape)

scaled_test_input = mms_for_x.transform(test_input)
final_test_input = scaled_test_input.reshape(1, 2, 1)
predicted = model.predict(final_test_input)
answer = mms_for_y.inverse_transform(predicted)
print(f"predicted: {answer}")
