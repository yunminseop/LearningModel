import random

S = list(range(0, 16))  # 상태
A = [0, 1, 2, 3]  # 행동

class Agent:
    def __init__(self):
        self.Q = {(s, a): 0 for s in S for a in A if s != 16}
        self.hole_state = [5, 7, 11, 12]
        self.n_episode = 10000
        self.Q[(15, -1)] = 1.0  # 목표 상태인 15에서는 -1의 행동을 하며 1.0의 가치를 가짐
        self.epsilon = 0.4 
        self.gamma = 0.7  # 할인율
        self.alpha = 0.9  # 학습률

    def transition(self, state, action):
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


    def epsilon_greedy(self, state):
        if state == 15:
            return 0
        
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


    def Q_learning(self, threshold=0.01):
        delta = float('inf')
        # while delta > threshold:
        for i in range(self.n_episode):
            delta = 0
            curr_state = 0

            while curr_state != 15:

                action = self.epsilon_greedy(curr_state)

                next_state = self.transition(curr_state, action)

                next_state = max(0, min(next_state, 15))  # 상태 0 이상, 15 이하로 제한

                # 보상 설정
                reward = self.get_reward(next_state)

                # Q값 업데이트
                old_q = self.Q[(curr_state, action)]

                self.Q[(curr_state, action)] += self.alpha * (
                    reward + self.gamma * max(self.Q.get((next_state, a), 0) for a in A) - old_q
                )

                delta = max(delta, abs(old_q - self.Q[(curr_state, action)]))

                # 상태 갱신
                curr_state = next_state
        
        del self.Q[(15,0)]
        del self.Q[(15,1)]
        del self.Q[(15,2)]
        del self.Q[(15,3)]

        for q in self.Q:
            print(f"Q{q}: {self.Q[q]}")

my_agent = Agent()
my_agent.Q_learning()