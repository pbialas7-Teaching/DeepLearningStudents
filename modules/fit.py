import numpy as np


def rmse(x, y, f):
    r = y - f(x)
    return np.sqrt(np.mean(r * r))


def fourier(x, y, n, lmb=0):
    S = np.sin(np.outer(np.arange(1, n), x))
    C = np.cos(np.outer(np.arange(0, n), x))
    B = np.concatenate((S @ y, C @ y))
    A = np.concatenate(
        (np.concatenate((S @ S.T, S @ C.T), 1), np.concatenate((C @ S.T, C @ C.T), 1)),
        0,
    )
    A += lmb * np.eye(2 * n - 1)
    sc = np.linalg.solve(A, B)

    def f(x):
        y = sc[: n - 1] @ np.sin(np.outer(np.arange(1, n), np.atleast_1d(x)))
        y += sc[n - 1 :] @ np.cos(np.outer(np.arange(0, n), np.atleast_1d(x)))
        return y

    return f
