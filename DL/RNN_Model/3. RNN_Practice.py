# 문제: 문자 시퀀스 예측 (RNN 활용)
# 주어진 텍스트 데이터를 이용하여, RNN을 사용해 그 다음에 올 문자를 예측하는 모델을 만들어보세요.

# 텍스트 데이터를 준비하세요.

# 예시: "hello world" (자유롭게 텍스트를 수정할 수 있습니다)
# 텍스트 데이터에서 각 문자를 고유한 숫자 인덱스로 매핑하세요.

# 각 시퀀스에서 주어진 문자를 입력으로 하고, 그 다음에 올 문자를 타겟 값으로 설정하여 학습 데이터를 준비하세요.

# RNN 모델을 구성하고 학습시키세요.

# 모델은 SimpleRNN 레이어를 사용하여 구현하고, 마지막 출력은 Dense 레이어로 설정하여 각 문자의 확률을 예측하도록 만듭니다.
# 훈련된 모델을 사용하여, 주어진 시퀀스에 대해 다음 문자를 예측하는 프로그램을 구현하세요.

# 예시: "hell" -> "o"
# 요구 사항:
# 모델: SimpleRNN + Dense (Softmax 활성화 함수)
# 학습: 텍스트에서 시퀀스 데이터를 만들고, 이를 RNN에 입력하여 모델을 훈련시킵니다.
# 예측: 훈련된 모델을 사용해 새로운 시퀀스에서 그 다음 문자를 예측합니다.
# 이 문제를 통해 RNN을 활용한 기본적인 시퀀스 예측을 실습할 수 있습니다!

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

sample_text = "gymnasium"
text_dict = {}

def mapping(sample_data):
    sorted_sample_data = sorted(set(sample_data))

    char_to_index = {each: idx for idx, each in enumerate(sorted_sample_data)}

    return char_to_index

def I2C(char_to_index, sample_data):
    converted_char= []
    for alphabet in sample_data:
        for item in char_to_index.items():
            if item[0] == alphabet:
                converted_char.append(str(item[1]))
    
    for i in range(len(converted_char)):
        converted_char[i] = int(converted_char[i])

    converted_char = np.asarray(converted_char)
    return converted_char

train_data_mapped = mapping(sample_text)
train_data = I2C(train_data_mapped, sample_text)

X = train_data[:-1]
X = X.reshape(-1, 1, 1)

# print(X.shape)
# print(type(X))

y = np.asarray(train_data[1:])
y = y.reshape(-1, 1)

# print(type(y))
# print(y.shape)

# model = Sequential([
#     SimpleRNN(10, activation = 'softmax', input_shape = (len(X), 1)),
#     Dense(1)
# ])

model = Sequential([
    SimpleRNN(10, activation='tanh', input_shape=(X.shape[1], 1)),
    Dense(len(train_data), activation='softmax')  # Softmax for multi-class classification
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=1000, batch_size=8)

test_text = "gym"

test_data_mapped = mapping(test_text)
test_data = I2C(test_data_mapped, test_text)

test_X = test_data[:-1]
test_X = test_X.reshape(-1, 1)
predicted = model.predict(test_X)
print(predicted)
# for item in test_data_mapped.items():
#     if item[1] == round(predicted[0][0]):
#         print(f"{predicted[0][0]:.2f}")
#         print(f"RNN 예측값: {item[0]}")
#         break

predicted_index = np.argmax(predicted)
print(f"예측 인덱스: {predicted_index}")
print(f"맵핑된 train_data: {train_data_mapped}")

for char, idx in train_data_mapped.items():
    if idx == predicted_index:
        print(f"gy{char}")
        
