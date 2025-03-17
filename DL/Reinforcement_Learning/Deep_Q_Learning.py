import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import random
import matplotlib.pyplot as plt
tf.config.run_functions_eagerly(True)

class DeepQLearningAgent:
    def __init__(self):
        self.state_size = 100
        self.action_size = 4
        self.gamma = 0.9 
        self.epsilon = 0.3 
        self.learning_rate = 0.001
        self.n_episodes = 500
        self.hole_state = [5, 7, 11, 12, 17, 22, 24, 29, 31, 36, 37, 39, 45, 50, 55, 60, 62, 69, 75, 81, 83, 86, 90, 93, 97]
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        q_values = self.model.predict(np.identity(self.state_size)[state].reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def train(self):
        for episode in range(self.n_episodes):
            print(f"Ep{episode} ====================================================")
            state = 0
            self.epsilon = max(0.1, 1 - (episode / self.n_episodes))
            
            while state != 99:
                action = self.epsilon_greedy(state)
                next_state = self.stochastic_transition(state, action)
                reward = self.get_reward(next_state)
                
                q_values = self.model.predict(np.identity(self.state_size)[state].reshape(1, -1), verbose=0)
                print(f"변경 전: {q_values}")
                print(f"행동: {action}")
                print(f"다음 상태: {next_state}")
                target = reward + self.gamma * np.max(self.model.predict(np.identity(self.state_size)[next_state].reshape(1, -1), verbose=0)) if next_state != 99 else reward
                q_values[0][action] = target
                print(np.identity(self.state_size)[state].reshape(1, -1))
                print(f"변경 후: {q_values}")
                
                self.model.fit(np.identity(self.state_size)[state].reshape(1, -1), q_values, epochs=1, verbose=1)
                
                state = next_state
                
                if state in [5, 7, 11, 12, 17, 22, 24, 29, 31, 36, 37, 39, 45, 50, 55, 60, 62, 69, 75, 81, 83, 86, 90, 93, 97, 99]:
                    break

        self.model.save("FL_Model_2.h5")
    
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
    
    def validation(self, name = None):
        model = load_model(name, custom_objects={'mse': MeanSquaredError()})
        op_list = []
        for i in range(self.state_size):
            optimal_policy = np.argmax(model.predict(np.identity(self.state_size)[i].reshape(1, -1)))
            match optimal_policy:
                case 0: op_list.append("Left")
                case 1: op_list.append("Down")
                case 2: op_list.append("Right")
                case 3: op_list.append("Up")
            
        print(f"op_list: {op_list}")
        
        grid_size = 10
        grid = np.zeros((grid_size, grid_size))

        fig, ax = plt.subplots(figsize=(5, 5))
        
        highlight_cells = []

        for hole in self.hole_state:
            highlight_cells.append(divmod(hole, 10))

        ax.set_xticks(np.arange(-0.5, grid_size, 1))
        ax.set_yticks(np.arange(-0.5, grid_size, 1))
        ax.imshow(grid, cmap='Blues', alpha=0.3)

       
        for i in range(grid_size):
            for j in range(grid_size):
                state = i * grid_size + j

                # Hole
                if (i, j) in highlight_cells:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightcoral', alpha=0.5))


                if state in self.hole_state:
                    ax.text(j, i, "Hole", ha='center', va='center', fontsize=11, fontweight='bold')
                elif state == 99:
                    ax.text(j, i, "Goal", ha='center', va='center', fontsize=11, fontweight='bold')
                else:
                    ax.text(j, i, op_list[state], ha='center', va='center', fontsize=11, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()


    def get_reward(self, state):
        if state == 99: return 10
        elif state in [5, 7, 11, 12, 17, 22, 24, 29, 31, 36, 37, 39, 45, 50, 55, 60, 62, 69, 75, 81, 83, 86, 90, 93, 97]: return -1
        return 3

agent = DeepQLearningAgent()
agent.train()
agent.validation(name="FL_Model_2.h5")