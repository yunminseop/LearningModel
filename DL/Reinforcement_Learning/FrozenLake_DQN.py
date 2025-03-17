import numpy as np
import tensorflow as tf
import random
import gym
from collections import deque
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

dis = 0.9
REPLAY_MEMORY = 50000

class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.state_size = 100
        self.action_size = 4
        self.gamma = 0.9 
        self.epsilon = 0.3 
        self.learning_rate = 0.001
        self.n_episodes = 500
        self.hole_state = [5, 7, 11, 12, 17, 22, 24, 29, 31, 36, 37, 39, 45, 50, 55, 60, 62, 69, 75, 81, 83, 86, 90, 93, 97]

        self.mainDQN = self.build_model()
        self.targetDQN = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, input_shape=(self.state_size,)),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def epsilon_greedy(self, state):
        print(self.mainDQN.summary())  # 모델 구조 출력
        state_array = np.identity(self.state_size)[state].reshape(1, -1)
        print(f"State shape: {state_array.shape}")
        print(state_array)

        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        q_values = self.mainDQN.predict(np.identity(self.state_size)[state].reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay_train(self, optimizer, loss_fn, train_batch):
        x_stack = []
        y_stack = []

        for state, action, reward, next_state, done in train_batch:
            q_pred = self.mainDQN.predict(np.identity(self.state_size)[state].reshape(1, -1), verbose=0)
            q_target = reward if done else reward + self.gamma * np.max(self.targetDQN.predict(np.identity(self.state_size)[next_state].reshape(1, -1), verbose=0))
            q_pred = np.array(q_pred)
            q_pred[0, action] = q_target

            x_stack.append(state)
            y_stack.append(q_pred[0])

        with tf.GradientTape() as tape:
            predictions = self.mainDQN(np.array(x_stack))
            loss = loss_fn(np.array(y_stack), predictions)
        grads = tape.gradient(loss, self.mainDQN.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.mainDQN.trainable_variables))

        return loss.numpy()

    def stochastic_transition(self, state, action):
        row, col = divmod(state, 10)
        if action == 0: 
            next_states = [row * 10 + max(0, col - 1), row * 10 + min(9, col + 1)] # 80% left, 20% right
        elif action == 1: 
            next_states = [min(9, row + 1) * 10 + col, max(0, row - 1) * 10 + col] # 80% down, 20% up
        elif action == 2: 
            next_states = [row * 10 + min(9, col + 1), row * 10 + max(0, col - 1)] # 80% right, 20% left
        elif action == 3: 
            next_states = [max(0, row - 1) * 10 + col, min(9, row + 1) * 10 + col] # 80% up, 20% down

        return random.choices(next_states, weights=[0.8, 0.2])[0]
    
    def get_reward(self, state):
        if state == 99: return 10
        elif state in [5, 7, 11, 12, 17, 22, 24, 29, 31, 36, 37, 39, 45, 50, 55, 60, 62, 69, 75, 81, 83, 86, 90, 93, 97]: return -5
        return 5
    
    def copy_weights(self):
        self.targetDQN.set_weights(self.mainDQN.get_weights())

    def bot_play(self):
        state = 0
        reward_sum = 0
        while True:
            action = self.epsilon_greedy(state)

            next_state = self.stochastic_transition(state, action)
            reward = self.get_reward(next_state)

            truncated = state in [5, 7, 11, 12, 17, 22, 24, 29, 31, 36, 37, 39, 45, 50, 55, 60, 62, 69, 75, 81, 83, 86, 90, 93, 97, 99]
            terminated = state in [99]

            reward_sum += reward
            state = next_state

            if truncated or terminated:
                print(f"Total Score: {reward_sum}")
                break

def main():
    
    max_episodes = 5000
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    agent = DQN()
    # mainDQN = DQN(input_size, output_size)
    # targetDQN = DQN(input_size, output_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    agent.copy_weights()

    for episode in range(max_episodes):
        state = 0
        step_count = 0
        done = False
        
        while not done:
            action = agent.epsilon_greedy(state)

            next_state = agent.stochastic_transition(state, action)
            reward = agent.get_reward(next_state)

            truncated = state in [5, 7, 11, 12, 17, 22, 24, 29, 31, 36, 37, 39, 45, 50, 55, 60, 62, 69, 75, 81, 83, 86, 90, 93, 97, 99]
            terminated = state in [99]

            if truncated or terminated:
                done = True
                break

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            step_count += 1

        print(f"Episode {episode} Step {step_count}")

        if episode % 10 == 1 and len(replay_buffer) >= 10:
            minibatch = random.sample(replay_buffer, 10)
            loss = agent.replay_train(optimizer, loss_fn, minibatch)
            print("Loss:", loss)
            agent.copy_weights()
            # copy_weights(mainDQN, targetDQN)

    agent.bot_play()

if __name__ == "__main__":
    main()