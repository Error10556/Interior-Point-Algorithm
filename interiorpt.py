import numpy as np
inv = np.linalg.inv


SOLVED = 1
UNSOLVABLE = 2
UNBOUNDED = 3


def interior_solve(objective, constraints, initialsol, alpha, eps):
    c = objective
    a = constraints
    x = initialsol
    solved = False
    eye = np.eye(c.shape[1])
    onevector = np.matrix('; '.join('1' * x.shape[0]))
    while not solved:
        diag = np.diag(x)
        a_ = a * diag
        c_ = diag * c
        a_t = a_.transpose()
        p = eye - a_t * inv(a_ * a_t) * a_
        cp = p * c_
        nu = max(-i for i in cp.A1)
        if nu <= eps:
            return (UNSOLVABLE,)
        newx = diag * (onevector + alpha / nu * cp)
        diff = newx - x
        absdiff = (diff.transpose() * diff).A1[0]
        if absdiff < eps:
            solved = True
        x = newx
        if (x.transpose() * x).A1[0] > 1e9:
            return (UNBOUNDED,)
    return (SOLVED, x)
