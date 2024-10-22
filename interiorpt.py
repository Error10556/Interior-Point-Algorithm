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
    eye = np.eye(c.shape[0])
    onevector = np.matrix('; '.join('1' * x.shape[0]))
    while not solved:
        diag = np.diag(x.A1)
        a_ = a * diag
        c_ = diag * c
        a_t = a_.transpose()
        p = eye - a_t * inv(a_ * a_t) * a_
        cp = p * c_
        nu = max(-i for i in cp.A1)
        if nu <= 0:
            return (UNSOLVABLE,)
        newx = diag * (onevector + alpha / nu * cp)
        diff = newx - x
        absdiff = max(abs(i) for i in diff.A1)
        if absdiff < eps:
            solved = True
        x = newx
        if (x.transpose() * x).A1[0] > 1e9:
            return (UNBOUNDED,)
    return (SOLVED, x)


def print_result_debug(tup):
    status = tup[0]
    if status == SOLVED:
        print(*map(lambda x: round(x, 3), tup[1].A1))
    elif status == UNBOUNDED:
        print("Unbounded!")
    elif status == UNSOLVABLE:
        print("Unsolvable!")


def main():
    xinit = np.matrix('2 2 4 3').T
    a = np.matrix('2 -2 8 0; -6 -1 0 -1')
    c = np.matrix('-2; 3; 0; 0')
    print_result_debug(interior_solve(c, a, xinit, 0.5, 0.0001))
    xinit = np.matrix("1;1;1;315;174;169")
    c = np.matrix('9;10;16;0;0;0')
    a = np.matrix('18 15 12 1 0 0; 6 4 8 0 1 0; 5 3 3 0 0 1')
    print_result_debug(interior_solve(c, a, xinit, 0.5, 0.0001))


if __name__ == "__main__":
    main()
