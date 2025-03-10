import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 💡 데이터 생성 (시계열 데이터)
def create_dataset(seq_length=4, num_samples=100):
    X, y = [], []
    for i in range(num_samples):
        seq = np.arange(i, i + seq_length + 1)  # 예: [0,1,2,3,4]
        X.append(seq[:-1])  # 입력: [0,1,2,3]
        y.append(seq[-1])   # 정답: 4
    return np.array(X), np.array(y)

# 데이터셋 준비
X, y = create_dataset()
X = X.reshape(X.shape[0], X.shape[1], 1)  # RNN 입력 형태 (samples, timesteps, features)

# 💡 RNN 모델 구성
model = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(4, 1)),  # RNN 레이어
    Dense(1)  # 출력 레이어
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, batch_size=8)

# 예측 테스트
test_input = np.array([[100, 101, 102, 103]])  # [100,101,102,103] → ?
test_input = test_input.reshape(1, 4, 1)  # (1, timesteps, features)
predicted = model.predict(test_input)
print(f"RNN 예측값: {predicted[0][0]:.2f}")
