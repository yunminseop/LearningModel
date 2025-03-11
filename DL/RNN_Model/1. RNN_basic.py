import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

def create_dataset(seq_length=4, num_samples=100):
    X, y = [], []
    for i in range(num_samples):
        seq = np.arange(i, i + seq_length + 1)  # 예: [0,1,2,3,4]
        X.append(seq[:-1])  # [0,1,2,3]
        y.append(seq[-1])   # 4
    return np.array(X), np.array(y)

X, y = create_dataset()
print(X.shape, y.shape)
""" X는 [[0, 1, 2, 3],
         [1, 2, 3, 4],
         [2, 3, 4, 5],
         ...        ]
         
         의 100 x 4 사이즈를 갖는다."""

X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, timesteps, features)

model = Sequential([
    SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(4, 1)),
    SimpleRNN(20, activation='relu', return_sequences=True), 
    SimpleRNN(10, activation='relu'), 
    Dense(1) 
])



model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=8)

test_input = np.array([[100, 101, 102, 103]])  # [100,101,102,103]
test_input = test_input.reshape(1, 4, 1)  # (1, timesteps, features)
predicted = model.predict(test_input)
print(f"RNN 예측값: {predicted[0][0]:.2f}")
