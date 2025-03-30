from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Bidirectional
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import r2_score



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def load_df():
    df = pd.read_csv(url)

    df.head()

    df.drop(index=[0, 1], inplace=True)

    df.reset_index(drop=True, inplace=True)

    df = df.astype(dtype={"Close":"float", "High":"float", "Low":"float", "Open":"float", "Volume":"int"}, errors="raise", copy=True)

    df.rename(columns={"Price":"Date"}, inplace=True)

    simple_dataset = np.array(df["Close"])

    return simple_dataset

class Model:
    def __init__(self):
        self.loss_func = MeanSquaredError()
        self.epoch_size = 1000
        self.batch_size = 8
        self.activation = "relu"
        self.model = Sequential([
            SimpleRNN(10, activation=self.activation, return_sequences=True, input_shape=(7, 1)),
            Bidirectional(LSTM(20, return_sequences=True)),
            SimpleRNN(10, activation=self.activation),
            Dense(1)
        ])

    # 일주일 분량의 데이터로 바로 다음날 가격 예측.   
    # Many to One   
    # 1월 2일 ~ 1월 8일 (7일간)의 데이터로 1월 9일 가격 예측   
    # 1월 3일 ~ 1월 9일 (7일간)의 데이터로 1월 10일 가격 예측   
    # ...

    def preprocess(self, simple_dataset):
        input_dim = 7

        x_train = []
        y_train = []
        for i in range(len(simple_dataset)):
            mini_x_list = []
            mini_y_list = []
            for j in range(input_dim):
                try:
                    mini_x_list.append(simple_dataset[i+j])
                    if j == input_dim - 1:
                        mini_y_list.append(simple_dataset[i+j+1])
                except:
                    pass
            x_train.append(mini_x_list)
            y_train.append(mini_y_list)

        x_train = np.array(x_train[:-7])
        y_train = np.array(y_train[:-7])

        X_train = pad_sequences(x_train, maxlen=7)
        X_train, X_temp, y_train, y_temp = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

        # temp를 다시 valid + test로 분할
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

        X_train = np.reshape(X_train, (-1, 7, 1))
        X_test = np.reshape(X_test, (-1, 7, 1))

        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_valid_reshaped = X_valid.reshape(X_valid.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

        X_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = X_scaler.fit_transform(X_train_reshaped)
        X_valid_scaled = X_scaler.transform(X_valid_reshaped)
        X_test_scaled = X_scaler.transform(X_test_reshaped)

        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_valid_scaled = y_scaler.transform(y_valid)
        y_test_scaled = y_scaler.transform(y_test)

        X_train_scaled = X_train_scaled.astype("float32")
        X_valid_scaled = X_valid_scaled.astype("float32")
        X_test_scaled = X_test_scaled.astype("float32")
        y_train_scaled = y_train_scaled.astype("float32")
        y_valid_scaled = y_valid_scaled.astype("float32")
        y_test_scaled = y_test_scaled.astype("float32")


        return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled, y_scaler
    
    def train(self, X_train, y_train):
        self.model.compile(optimizer='adam', loss=self.loss_func, metrics=["accuracy", "RootMeanSquaredError"])
        self.model.fit(X_train, y_train, epochs=self.epoch_size, batch_size=self.batch_size)
        self.model.save("Finance_Model.h5")

    def eval(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse}")

    def summary(self):
        print(self.model.summary())

    def use_model(self):
        print("모델 불러오는 중...")
        model = load_model('Finance_Model.h5')
        return model


url = "/home/ms/ws/git_ws/LearningModel/DL/RNN_Model/samsung_stock.csv"
df = load_df()
model = Model()
X_train, X_valid, X_test, y_train, y_valid, y_test, y_scaler = model.preprocess(df)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

# model.train(X_train, y_train)
saved_model = model.use_model()
# results = saved_model.evaluate(X_valid, y_valid)
# print(results)
model.summary()

test_pred = saved_model.predict(X_test)
test_pred_it = y_scaler.inverse_transform(test_pred)
test_label_it = y_scaler.inverse_transform(y_test)

for p, l  in zip(test_pred_it, test_label_it):
    print(f"예측: {p}, 정답: {l}")

mse = mean_squared_error(y_test, test_pred)
print("MSE:", mse)

r2 = r2_score(y_test, test_pred)
print("R² Score:", r2)