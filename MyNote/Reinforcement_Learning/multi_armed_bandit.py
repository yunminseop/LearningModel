import numpy as np

arms_profit = [0.4, 0.12, 0.52, 0.6, 0.25]
n_arms = len(arms_profit)

n_trial = 1000

def pull_bandit(handle):
    q = np.random.random()
    if q <arms_profit[handle]:
        return 1
    else:
        return -1
    
""" 1. random policy """

# def random_exploration():
#     episode = []
#     num = np.zeros(n_arms)
#     wins = np.zeros(n_arms)

#     for i in range(n_trial):
#         h = np.random.randint(0, n_arms) # 선택하는 손잡이(handle)의 번호 0~4번.
#         reward = pull_bandit(h) # 핸들을 당겨 얻은 결과

#         episode.append([h, reward]) # 몇 번 핸들을 눌러 무슨 보상을 받았는지 기록
#         num[h] += 1 # 해당 핸들의 누른 횟수 증가
#         wins[h] += 1 if reward == 1 else 0 # 보상이 1이면 승리 횟수 추가, 아니면 추가 안 함.

#     return episode, (num, wins)

# e, r = random_exploration() # e = 에피소드 전체 리스트, r = 손잡이별 누른 횟수와 승리 횟수 반환

# print("손잡이별 승리 확률:", ["%6.4f"% (r[1][i]/r[0][i]) if r[0][i] > 0 else 0.0 for i in range(n_arms)])
# print("손잡이별 수익:($):", ["%d"% (2*r[1][i]-r[0][i]) for i in range(n_arms)])
# print("순 수익($):", sum(np.asarray(e)[:,1]))


""" 2. epsilon_greedy algorithm
 - 수학적 현상을 난수를 생성하여 시뮬레이션하는 기법 '몬테 카를로 방법' 
 - 기본적으로는 탐욕 알고리즘이지만 epsilon(이하 eps)의 비율만큼 탐험을 적용하여 탐험과 탐사의 균형을 추구 """

def epsilon_greedy(eps):
    episode = []
    num = np.zeros(n_arms)
    wins = np.zeros(n_arms)

    for i in range(n_trial):
        r = np.random.random()
        if (r<eps or sum(wins)==0):
            h = np.random.randint(0, n_arms)
        else:
            prob = np.asarray([wins[i]/num[i] if num[i]>0 else 0.0 for i in range(n_arms)])
            prob = prob/sum(prob)
            h = np.random.choice(range(n_arms), p=prob)
        reward = pull_bandit(h)
        episode.append([h, reward])
        num[h]+=1
        wins[h]+=1 if reward==1 else 0
    return episode, (num, wins)

e, r = epsilon_greedy(0.1)

print("손잡이별 승리 확률:", ["%6.4f"% (r[1][i]/r[0][i]) if r[0][i] > 0 else 0.0 for i in range(n_arms)])
print("손잡이별 수익:($):", ["%d"% (2*r[1][i]-r[0][i]) for i in range(n_arms)])
print("순 수익($):", sum(np.asarray(e)[:,1]))