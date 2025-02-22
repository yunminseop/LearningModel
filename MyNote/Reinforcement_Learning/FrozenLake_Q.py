"""FrozenLake -> Q_Learning"""
import numpy as np
import random
import matplotlib.pyplot as plt


S = list(range(0, 16))  # 상태
A = [0, 1, 2, 3]  # 행동

class Agent:
    def __init__(self):
        self.Q = {(s, a): 0 for s in S for a in A if s != 16}
        self.hole_state = [5, 7, 11, 12]
        self.n_episode = 1000000
        self.Q[(15, -1)] = 1.0  # 목표 상태인 15에서는 -1의 행동을 하며 1.0의 가치를 가짐
        self.epsilon = 0.4 
        self.gamma = 0.99  # 할인율
        self.alpha = 0.0  # 학습률

        self.total_state = {each:[] for each in S}
        self.optimal_policy = {each: 0 for each in S}


    def deterministic_transition(self, state, action):
        row, col = divmod(state, 4)  # 현재 상태를 (row, col)로 변환

        if action == 0:  # 왼쪽으로 이동
            col = max(0, col - 1)
        elif action == 1:  # 아래로 이동
            row = min(3, row + 1)
        elif action == 2:  # 오른쪽으로 이동
            col = min(3, col + 1)
        elif action == 3:  # 위로 이동
            row = max(0, row - 1)

        return row * 4 + col  # 다시 상태 번호로 변환
    

    def stochastic_transition(self, state, action):
        row, col = divmod(state, 4)

        if action == 0:  # 왼쪽으로 이동
            next_states = [row * 4 + max(0, col - 1), row * 4 + min(3, col + 1)]  # 왼쪽, 오른쪽
            weights = [0.8, 0.2]  # 왼쪽으로 갈 확률 80%, 오른쪽으로 갈 확률 20%

        elif action == 1:  # 아래로 이동
            next_states = [min(3, row + 1) * 4 + col, max(3, row - 1) * 4 + col]  # 아래, 위
            weights = [0.8, 0.2]  # 아래로 갈 확률 100%

        elif action == 2:  # 오른쪽으로 이동
            next_states = [row * 4 + min(3, col + 1), row * 4 + max(3, col + 1)]  # 오른쪽, 왼쪽
            weights = [0.8, 0.2]  # 오른쪽으로 갈 확률 100%

        elif action == 3:  # 위로 이동
            next_states = [max(0, row - 1) * 4 + col, max(0, row + 1) * 4 + col]  # 위, 아래
            weights = [0.8, 0.2]  # 위로 갈 확률 100%

        # weights와 population의 길이가 같아야 함
        next_state = random.choices(next_states, weights=weights, k=1)[0]
        return next_state



    def epsilon_greedy(self, state):
        if state == 15:
            return -1
        
        elif random.random() < self.epsilon:
            return random.choice(A)  # 무작위 행동

        else:
            q_values = [self.Q.get((state, a), 0) for a in A]
            max_q = max(q_values)
            return A[q_values.index(max_q)]  # Q값이 가장 큰 행동을 선택
        

    def get_reward(self, state):
        if state == 15:  # 목표 상태
            return 1
        elif state in [5, 7, 11, 12]:  # 구멍 위치
            return -1
        return 0  # 나머지 보상 없음


    def is_done(self, state):
        return state == 15 or state in [5, 7, 11, 12]  # 목표 지점 or 구멍


    def Q_learning(self):
        cnt = 0
        for _ in range(self.n_episode):
            curr_state = 0
            cnt += 1
            self.alpha = max(0.00015, 1 / (0.001 * cnt + 1))
            self.epsilon = max(0.1, 1 - (cnt / self.n_episode))
            print(self.alpha, self.epsilon)
            
            while curr_state != 15:

                action = self.epsilon_greedy(curr_state)

                next_state = self.stochastic_transition(curr_state, action)

                next_state = max(0, min(next_state, 15))  # 상태 0 이상, 15 이하로 제한

                # 보상 설정
                reward = self.get_reward(next_state)

                # Q값 업데이트
                old_q = self.Q[(curr_state, action)]

                self.Q[(curr_state, action)] += self.alpha * (
                    reward + self.gamma * max(self.Q.get((next_state, a), 0) for a in A) - old_q
                )

                # 상태 갱신
                curr_state = next_state

                if self.is_done(curr_state): # 다음 상태가 구멍이거나 목표 상태이면 이번 반복문은 여기서 종료
                    break
        
        del self.Q[(15,0)]
        del self.Q[(15,1)]
        del self.Q[(15,2)]
        del self.Q[(15,3)]


    def show_optimal_policy(self):
        
        for state in S:
            for key in self.Q.keys():
                if state == key[0]:
                    self.total_state[state].append(self.Q[key])

        for item in self.total_state.items():
            match np.argmax(item[1]):
                case 0: self.optimal_policy[item[0]] = "←"
                case 1: self.optimal_policy[item[0]] = "↓"
                case 2: self.optimal_policy[item[0]] = "→"
                case 3: self.optimal_policy[item[0]] = "↑"

        print(self.optimal_policy)

        grid_size = 4
        grid = np.zeros((grid_size, grid_size))

        fig, ax = plt.subplots(figsize=(5, 5))
        
        highlight_cells = []

        for hole in self.hole_state:
            highlight_cells.append(divmod(hole, 4))

        ax.set_xticks(np.arange(-0.5, grid_size, 1))
        ax.set_yticks(np.arange(-0.5, grid_size, 1))
        ax.imshow(grid, cmap='Blues', alpha=0.3)

       
        for i in range(grid_size):
            for j in range(grid_size):
                state = i * grid_size + j

                # Hole
                if (i, j) in highlight_cells:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightcoral', alpha=0.5))

                # Policy
                if state in self.optimal_policy:
                    if state in self.hole_state:
                        ax.text(j, i, "Hole", ha='center', va='center', fontsize=14, fontweight='bold')
                    elif state == 15:
                        ax.text(j, i, "Goal", ha='center', va='center', fontsize=14, fontweight='bold')
                    else:
                        ax.text(j, i, self.optimal_policy[state], ha='center', va='center', fontsize=16, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

                    



my_agent = Agent()
my_agent.Q_learning()
my_agent.show_optimal_policy()