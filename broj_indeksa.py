def domaci_1(broj, godina=None):
    s = 0
    while broj:
        s += broj % 10
        broj //= 10

    zadaci = {
        0: 'grebena(ridge) regresija',
        1: 'LASSO regresija',
        2: 'lokalno ponderisana linearna regresija'
    }

    ind = s % 3
    return ind, zadaci[ind]


if __name__ == '__main__':
    print(domaci_1(3140))
