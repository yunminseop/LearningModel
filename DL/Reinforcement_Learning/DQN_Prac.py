# import numpy as np
# import tensorflow as tf
# import random
# import dqn
# from collections import deque
# tf.compat.v1.disable_eager_execution()
# import gym
# env = gym.make("CartPole-v0")

# input_size = env.observation_space.shape[0]
# print(input_size)
# output_size = env.action_space.n
# print(output_size)
# dis = 0.9
# REPLAY_MEMORY = 50000

# def replay_train(mainDQN, targetDQN, train_batch):
#     x_stack = np.empty(0).reshape(0, input_size)
#     y_stack = np.empty(0).reshape(0, output_size)

#     for state, action, reward, next_state, done in train_batch:
#         Q = mainDQN.predict(state)


#         if done:
#             Q[0, action] = reward
#         else:
#             Q[0, action] = reward + dis*np.max(targetDQN.predict(next_state))

#         y_stack = np.vstack([y_stack, Q])
#         x_stack = np.vstack([x_stack, state])

#     return mainDQN.update(x_stack, y_stack)

# def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
#     op_holder = []

#     src_vars = tf.compat.v1.get_collection(
#         tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
#     dest_vars = tf.compat.v1.get_collection(
#         tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
#     for src_vars, dest_vars in zip(src_vars, dest_vars):
#         op_holder.append(dest_vars.assign(src_vars.value()))
    
#     return op_holder

# def bot_play(mainDQN):
#     s = env.reset()
#     reward_sum = 0
#     while True:
#         env.render(mode="human")
#         a = np.argmax(mainDQN.predict(s))
#         s, reward, done, _ = env.step(a)
#         reward_sum += reward
#         if done:
#             print(f"Total score: {reward_sum}")
#             break

# def main():
#     max_episodes = 5000
#     replay_buffer = deque()

#     with tf.compat.v1.Session() as sess:
#         mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
#         targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
#         tf.compat.v1.global_variables_initializer().run()

#         copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

#         sess.run(copy_ops)

#         for episode in range(max_episodes):
#             e = 1. / ((episode/10)+1)
#             done = False
#             step_count = 0
#             state = env.reset()[0]
            
#             while not done:
#                 if np.random.rand(1) < e:
#                     action = env.action_space.sample()
#                 else:
#                     action = np.argmax(mainDQN.predict(state))

#                 next_state, reward, terminated, truncated, _ = env.step(action)

#                 if terminated or truncated:
#                     done = True
#                 else:
#                     done = False

#                 if truncated:
#                     reward -= 100

#                 replay_buffer.append((state, action, reward, next_state, done))
#                 if len(replay_buffer) > REPLAY_MEMORY:
#                     replay_buffer.popleft()

#                 state = next_state
#                 step_count += 1
#                 if step_count > 10000:
#                     break

#             print(f"Episode {episode} Step {step_count}")
#             if step_count > 10000:
#                 pass

#             if episode % 10 == 1:
#                 for _ in range(50):
#                     minibatch = random.sample(replay_buffer, 10)
#                     loss, _ = replay_train(mainDQN, targetDQN, minibatch)

#                 print("Loss:", loss)
#                 sess.run(copy_ops)

#         bot_play(mainDQN)

# if __name__ == "__main__":
#     main()


import numpy as np
import tensorflow as tf
import random
import gym
from collections import deque

# Gym 환경 설정
env = gym.make("CartPole-v1")
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
dis = 0.9  # 할인율
REPLAY_MEMORY = 50000

class DQN(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(24, activation='relu')
        self.fc2 = tf.keras.layers.Dense(24, activation='relu')
        self.fc3 = tf.keras.layers.Dense(output_size, activation=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

def replay_train(mainDQN, targetDQN, optimizer, loss_fn, train_batch):
    x_stack = []
    y_stack = []

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN(np.array([state]))
        Q_target = reward if done else reward + dis * np.max(targetDQN(np.array([next_state])))
        Q = np.array(Q)
        Q[0, action] = Q_target

        x_stack.append(state)
        y_stack.append(Q[0])

    with tf.GradientTape() as tape:
        predictions = mainDQN(np.array(x_stack))
        loss = loss_fn(np.array(y_stack), predictions)
    grads = tape.gradient(loss, mainDQN.trainable_variables)
    optimizer.apply_gradients(zip(grads, mainDQN.trainable_variables))

    return loss.numpy()

def copy_weights(mainDQN, targetDQN):
    targetDQN.set_weights(mainDQN.get_weights())

def bot_play(mainDQN):
    state, _ = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = np.argmax(mainDQN(np.array([state])))
        next_state, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward
        state = next_state
        if terminated or truncated:
            print(f"Total Score: {reward_sum}")
            break

def main():
    max_episodes = 5000
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    mainDQN = DQN(input_size, output_size)
    targetDQN = DQN(input_size, output_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    copy_weights(mainDQN, targetDQN)

    for episode in range(max_episodes):
        epsilon = 1.0 / ((episode / 10) + 1)
        state, _ = env.reset()
        step_count = 0
        done = False
        
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(mainDQN(np.array([state])))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if truncated:
                reward -= 100

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            step_count += 1

        print(f"Episode {episode} Step {step_count}")

        if episode % 10 == 1 and len(replay_buffer) >= 10:
            minibatch = random.sample(replay_buffer, 10)
            loss = replay_train(mainDQN, targetDQN, optimizer, loss_fn, minibatch)
            print("Loss:", loss)
            copy_weights(mainDQN, targetDQN)

    bot_play(mainDQN)

if __name__ == "__main__":
    main()