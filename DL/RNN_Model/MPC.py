import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

def compute_optimal_control(v_pred, delta_pred, target_speed, target_steering_angle):
    """
    미래 예측 값을 기반으로 최적의 제어 입력을 계산하는 함수
    :param v_pred: 예측된 미래 속도
    :param delta_pred: 예측된 미래 조향각
    :param target_speed: 목표 속도
    :param target_steering_angle: 목표 조향각
    :return: 최적 제어 입력
    """
    speed_error = v_pred - target_speed
    
    steering_error = delta_pred - target_steering_angle
    
    optimal_control = -0.1 * speed_error - 0.5 * steering_error
    
    return optimal_control


def build_mpc_model(time_horizon=10, image_shape=(128, 128, 3)):

    img = Input(shape=image_shape, name='image_input')  # 이미지
    velocity = Input(shape=(1,), name='speed_input')    # 속도
    steering = Input(shape=(1,), name='steering_input') # 조향각

    conv = layers.Conv2D(16, (5, 5), strides=2, activation='relu')(img)
    conv = layers.Conv2D(32, (5, 5), strides=2, activation='relu')(conv)
    conv = layers.Conv2D(64, (3, 3), strides=2, activation='relu')(conv)
    conv = layers.GlobalAveragePooling2D()(conv)

    concat = layers.Concatenate()([conv, velocity, steering])

    dense = layers.Dense(256, activation='relu')(concat)
    dense = layers.Dense(128, activation='relu')(dense)
    output = layers.Dense(time_horizon * 2)(dense) 

    v_pred = layers.Lambda(lambda x: x[:, :time_horizon], name='v_pred')(output)
    s_pred = layers.Lambda(lambda x: x[:, time_horizon:], name='s_pred')(output)

    model = models.Model(inputs=[img, velocity, steering], outputs=[v_pred, s_pred])
    return model


def build_model(image_shape=(128, 128, 3)):
    img_input = Input(shape=image_shape, name='img_input')
    velocity_input = Input(shape=(1,), name='vel_input')
    steering_input = Input(shape=(1,), name='steer_input')

    mpc_model = build_mpc_model(time_horizon=10, image_shape=image_shape)
    v_pred, s_pred = mpc_model([img_input, velocity_input, steering_input])

    x = layers.Concatenate(axis=-1)([v_pred, s_pred])

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(2, name='current_prediction')(x)

    model = models.Model(inputs=[img_input, velocity_input, steering_input], outputs=output, name="final_model")
    return model



if __name__=="__main__":
    window_size = 10
    
    model = build_model()
    model.summary()

    img_input = np.random.randn(window_size, 128, 128, 3).astype(np.float32)  # 이미지 시퀀스
    velocity_input = np.random.randn(window_size, 1).astype(np.float32)       # 속도 시퀀스
    steering_input = np.random.randn(window_size, 1).astype(np.float32)       # 조향각 시퀀스

    # BATCH_SIZE = 8
    # train_size = len(train_can_arrays)
    # valid_size = len(valid_can_arrays)

    # def generator(can_dataset, image_dataset, label, input_dim=10, start=0, end=None):
    #     if end is None:
    #         end = len(can_dataset) - (input_dim +10)
    #     for i in range(start, end):
    #         x = can_dataset[i:i+input_dim]
    #         img = image_dataset[i:i+input_dim]
    #         y = label[i+input_dim:i+input_dim+10]
    #         y = y.reshape(-1, 1)
    #         yield (img, x), y

    # train_dataset = tf.data.Dataset.from_generator(
    #     lambda: generator(train_can_arrays, train_image_arrays, train_labels, input_dim=10, start=0),
    #     output_signature=(
    #         (
    #             tf.TensorSpec(shape=(10, 90, 424, 3), dtype=tf.float32),
    #             tf.TensorSpec(shape=(10, 50), dtype=tf.float32)
    #         ),
    #         tf.TensorSpec(shape=(10, 1), dtype=tf.float32)
    #     )
    # )

    # valid_dataset = tf.data.Dataset.from_generator(
    #     lambda: generator(valid_can_arrays, valid_image_arrays, valid_labels, input_dim=10, start=0),
    #     output_signature=(
    #         (
    #             tf.TensorSpec(shape=(10, 90, 424, 3), dtype=tf.float32),
    #             tf.TensorSpec(shape=(10, 50), dtype=tf.float32)
    #         ),
    #         tf.TensorSpec(shape=(10, 1), dtype=tf.float32)
    #     )
    # )

    # train_dataset = train_dataset.repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # valid_dataset = valid_dataset.repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # for x, y in train_dataset.take(3).as_numpy_iterator():
    #     print("x1:", x[0].shape)
    #     print("x2:", x[1].shape)
    #     print("y:", y.shape)

    # for x, y in valid_dataset.take(3).as_numpy_iterator():
    #     print("val_x1:", x[0].shape)
    #     print("val_x2:", x[1].shape)
    #     print("val_y:", y.shape)

    # print("=============================")

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # model.compile(optimizer=optimizer, loss='mse', metrics=['mae']) # 첫 모델 학습 시엔 r2_score라고 작성

    # steps_per_epoch = (train_size - 20) // BATCH_SIZE
    # validation_steps = (valid_size - 20) // BATCH_SIZE

    # # early_stop = EarlyStopping(
    # #     monitor='val_loss',   # 검증 손실을 기준으로 모니터링
    # #     patience=3,           # 개선이 없더라도 3 epoch 더 기다림
    # #     restore_best_weights=True  # 가장 성능 좋았던 가중치 복원
    # # )

    # history = model.fit(
    #     train_dataset,
    #     validation_data=valid_dataset,
    #     epochs=10,
    #     steps_per_epoch=steps_per_epoch,
    #     validation_steps=validation_steps,
    # )


    # print("v_pred.shape:", v_pred.shape)         # (4, 10)
    # print("s_pred.shape:", s_pred.shape)     # (4, 10)
    # print(f"v_pred: {v_pred}")
    # print(f"s_pred: {s_pred}")

    # compute_optimal_control(v_pred, s_pred, )