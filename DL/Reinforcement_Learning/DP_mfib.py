import numpy as np

"""memorized Fibonacci Sequence"""
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

"""iterative Fibonacci Sequence"""
def fib(n):
    fibo = []
    for i in range(n):
        if i <= 1:
            fibo.append(1)
        else:
            res = fibo[i-1] + fibo[i-2]
            fibo.append(res)
    return fibo

print(fib(10))

"""space-optimized Fibonacci"""
def fib2(n):
    first = 0
    second = 0
    for i in range(n):
        if i == 0:
            first = 1
            second = 0
        elif i == 1:
            first = 1
            second = 1
        else:
            first, second = first + second, first
        print(first, end=', ')
    return first

print(fib2(5)) 