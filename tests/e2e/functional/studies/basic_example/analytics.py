def g(k, L, p_max=250):
    return max(min(L - k * p_max, p_max), 0)


def nuclear_1(L):
    return min(L, 250)


def nuclear_2(L):
    return max(min(L - 250, 250), 0)


def gas_1(L):
    return max(min(L - 500, 250), 0)


def gas_2(L):
    return max(min(L - 750, 250), 0)


def nuclear(L):
    return min(L, 500)


def gas(L):
    return max(L - 500, 0)


def britishnuke(L):
    return max(min(L - 250, 250), 0)


def rhonepower(L):
    return min(L, 250) + max(L - 500, 0)
