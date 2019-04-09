import math


def nCr(n, r):
    f = math.factorial
    print(n)
    print(r)
    print(f(n) / f(r) / f(n-r))
    return f(n) / f(r) / f(n-r)


def code(k, n):
    sum = 0
    if n > k:

        return 0

    elif(n == k):

        return math.factorial(k)

    elif(n == 1 and k >= 0):

        return (2 ** k)-1

    else:
        for j in range(1, k - (n-1)+1):
            sum = sum + (nCr(k, j)) * code(k-j, n-1)
    return sum


print(code(15, 8))
