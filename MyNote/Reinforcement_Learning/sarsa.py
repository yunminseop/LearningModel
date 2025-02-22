import random

S = [0, 1, 2, 3]  # 상태
A = [-1, 1]  # 행동
R = [0, 1]  # 보상 (3에 도달하면 보상 1, 그 외엔 0)

class Agent:
    def __init__(self):
        self.Q = {(s, a): 0 for s in S for a in A if s != 3}
        self.Q[(3, 0)] = 1  # 상태 3에서는 행동 0의 Q값을 1로 설정
        self.epsilon = 0.4  # 탐욕적인 정책을 위한 epsilon
        self.gamma = 0.3  # 할인율
        self.alpha = 0.9  # 학습률

    def epsilon_greedy(self, state):
        if state == 3:
            return 0
        elif random.random() < self.epsilon:
            if state == 0:
                return 1
            return random.choice(A)  # 무작위 행동
        else:
            q_values = [self.Q.get((state, a), 0) for a in A]
            max_q = max(q_values)
            return A[q_values.index(max_q)]  # Q값이 가장 큰 행동을 선택

    def sarsa(self, threshold=0.01):
        delta = float('inf')
        while delta > threshold:
            delta = 0
            curr_state = 0
            action = self.epsilon_greedy(curr_state)

            while curr_state != 3:
                next_state = curr_state + action
                next_state = max(0, min(next_state, 3))  # 상태는 0 이상, 3 이하로 제한

                # 보상 설정
                reward = R[1] if next_state == 3 else R[0]

                # 다음 상태에서의 행동 선택
                next_action = self.epsilon_greedy(next_state)

                # Q값 업데이트
                old_q = self.Q[(curr_state, action)]

                self.Q[(curr_state, action)] += self.alpha * (
                    reward + self.gamma * self.Q.get((next_state, next_action), 0) - old_q
                )


                delta = max(delta, abs(old_q - self.Q[(curr_state, action)]))

                # 상태 갱신
                curr_state = next_state
                action = next_action

        self.Q.pop((0, -1))
        print(f"Q: {self.Q}")

my_agent = Agent()
my_agent.sarsa()
