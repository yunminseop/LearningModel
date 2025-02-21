import numpy as np
import random

""" 상태 전이 확률 분포 = 모두 1.0 (결정론적 환경)
    상태 1, 2 에서 좌or우로 이동할 확률 = 0.5 (랜덤 정책, 몬테카를로 방법)"""

S = [0, 1, 2, 3] # 상태. 0, 1, 2, 3

A = [-1, 1] # 행동. 왼쪽 이동: -1, 오른쪽 이동: 1

R = [0, 1] # 보상. 상태 3에 도착 시 보상: 1, 나머지 보상: 0

class agent:
    def __init__(self):
        self.state = S[0]
        self.gamma = 0.9
        self.value_function = {s: 0 for s in S} # 최적 가치 딕셔너리
        self.best_action = {s: 0 for s in S} # 최적 정책 딕셔너리
        self.action = None
        self.reward = None
        self.next_state = None
        self.value = None
    
    # 상태 전이 함수
    def transfer(self):
        p = random.random()

        if self.next_state is not None:
            self.state = self.next_state

        if self.state == S[0]: # state = 0 일 때, 
            self.action = A[1]; self.reward = R[0] # 오른쪽 이동 밖에 안되고 보상은 없다.

        elif self.state == S[1] or self.state == S[2]: # state = 1 or 2 일 때,
            if p >= 0.5:    # 0.5의 확률로 오른쪽이나 왼쪽으로 이동하며
                self.action = A[0]
            else: self.action = A[1]
            
            self.reward = R[0] # 보상은 없다.


        elif self.state == S[3]: # state = 3 일 때,
            self.action = 0 # 이동하지 않으며 (예외 처리)
            self.reward = R[1] # 보상이 있다.

        self.next_state = self.state + self.action

        return(self.state, self.action, self.reward, self.next_state)

    # 가치 반복 알고리즘
    def value_iteration(self, threshold=1e-6):
        delta = float('inf')
        while delta > threshold:
            delta = 0
            for s in S: # 각 상태에 대해서 반복문 실행
                v = self.value_function[s]  # 현재 상태의 가치 함수 저장
                max_value = float('-inf')
                
                for a in A: # 각 행동에 대해서 반복문 실행
                    next_state = s + a
                    if next_state not in S:  # 유효하지 않은 상태는 무시
                        continue
                    
                    # 벨만 최적 방정식에 따른 가치 계산
                    expected_value = R[0] + self.gamma * self.value_function.get(next_state, 0) if next_state != 3 else R[1]
                    
                    # 최적 행동
                    if expected_value > max_value:
                        if s == 3:
                            self.best_action[s] = '정지'
                        else:
                            if a == -1:
                                self.best_action[s] = '좌측 이동'
                            else:
                                self.best_action[s] = '우측 이동'

                    # 최적 가치 찾기
                    max_value = max(max_value, expected_value)
                    """ 최적 가치 함수란, 최적의 행동을 했을 때 이후 상태에서 얻을 수 있는 '기대 보상'을 반환함. 최적의 행동이 무엇인지는 모름. 왜냐?
                        행동(-1과 1)을 순회하며 각 행동에 대한 기대 보상을 구하고, 이들 중 최대 보상을 max_value에 저장하기 때문에 이것만으로는 어떤 행동이 최적 행동인지는 알 수 없음.
                        각 상태에서의 최적 행동을 알고 싶다면 각 행동에 대해 순회하는 동안 계산된 각각의 가치를 비교하여 높은 가치가 계산될 때마다 최적의 행동을 저장하면 됨."""
                    

                
                # 가치 함수 업데이트
                if s == 3:
                    self.value_function[s] = 1
                else:
                    self.value_function[s] = max_value
                delta = max(delta, abs(v - self.value_function[s]))  # 가치 변화 확인

        return self.value_function, self.best_action  # 수정된 가치 함수 반환



my_agent = agent()

# 상태 전이
while (my_agent.state != 3):
    res = my_agent.transfer()
    print(f"상태: {res[0]}, 행동: {res[1]}, 보상: {res[2]}, 다음 상태: {res[3]}")

# 가치 반복
value_function = my_agent.value_iteration()
print(f"최적 가치 함수: {value_function[0]}")
print(f"최적 정책: {value_function[1]}")
