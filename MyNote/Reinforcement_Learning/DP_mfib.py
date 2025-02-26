import numpy as np

def mfib(memo, n):
    if memo[n-1] != 0:
        return memo[n-1]
    elif n <= 2:
            memo[n-1] = 1
            return memo[n-1]
    else:
        memo[n-1] = mfib(memo,n-1) + mfib(memo,n-2)
        return memo[n-1]

n = 10
memo = np.zeros(n)

print(mfib(memo, n))
print(memo)