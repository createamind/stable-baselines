import numpy as np

def rec(x):
    x = x - 1
    print(x)
    if x < -10:
        return "stop"
    rec(x)


def list2int(x):
    num = ""
    for i in x:
        num += str(i)
    return int(num)


def mul(x, y):
    if type(x) is int:
        x = [int(x) for x in str(x)]
        y = [int(y) for y in str(y)]
    print(x, y)
    l = len(x)
    if l <= 1:
        return x[0] * y[0]
    else:
        a = list2int(x[:int(l / 2)])
        b = list2int(x[int(l / 2):])
        c = list2int(y[:int(l / 2)])
        d = list2int(y[int(l / 2):])
        bd = mul(b, d)
        ac = mul(a, c)
        ab_cd = mul(a + b, c + d)
        return (10 ** l) * ac + 10 ** int(l / 2) * (ab_cd - ac - bd) + bd


if __name__ == "__main__":
    x = 1234
    y = 2345

    re = mul(x, y)
    print("result", re)