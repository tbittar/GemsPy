def g(k, LOAD, p_max=250):
    return max(min(LOAD - k * p_max, p_max), 0)


def nuclear_1(LOAD):
    return min(LOAD, 250)


def nuclear_2(LOAD):
    return max(min(LOAD - 250, 250), 0)


def gas_1(LOAD):
    return max(min(LOAD - 500, 250), 0)


def gas_2(LOAD):
    return max(min(LOAD - 750, 250), 0)


def nuclear(LOAD):
    return min(LOAD, 500)


def gas(LOAD):
    return max(LOAD - 500, 0)


def britishnuke(LOAD):
    return max(min(LOAD - 250, 250), 0)


def rhonepower(LOAD):
    return min(LOAD, 250) + max(LOAD - 500, 0)
