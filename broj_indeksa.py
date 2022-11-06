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


def domaci_2(broj, godina=None):
    b = []

    while broj:
        b.append(broj % 10)
        broj //= 10

    b = b[::-1]

    if b[2] % 2 == 0:
        glm = 'LR'
    else:
        glm = 'softmax'

    if b[3] % 2 == 0:
        ga = 'GDA'
    else:
        ga = 'GNB'

    return b, glm, ga


if __name__ == '__main__':
    broj = 3140
    print('Domaci 1: ', domaci_1(broj))
    print('Domaci 2: ', domaci_2(broj))
