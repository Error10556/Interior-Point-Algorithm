import numpy as np
inv = np.linalg.inv


SOLVED = 1
UNSOLVABLE = 2
UNBOUNDED = 3

DEFAULT_EPS = 1e-7


def vectorAsTuple(vec):
    return tuple(round(i, 3) for i in (vec.A1 if
                 isinstance(vec, np.matrix) else vec))


def get_best(array, condition=lambda x: True):
    filtered = [i for i in range(len(array)) if condition(array[i])]
    if not filtered:
        return None
    return min(filtered, key=lambda i: array[i])


def normalize_row(mat, row, col):
    mat[row] /= mat[row, col]
    pass


def add_row(mat, source_row, target_row, multiplier):
    mat[target_row] += mat[source_row] * multiplier


def simplex(mat, eps):
    h, w = mat.shape
    fracs = [0] * (h - 1)
    basic = [w - h + i for i in range(h)]

    while True:
        col = get_best(list(mat[0].A1), lambda x: x < -eps)
        if col is None or mat[0, col] > -eps:
            break

        for i in range(h - 1):
            sol = mat[i + 1, -1]
            divisor = mat[i + 1, col]
            if sol >= 0 and sol <= eps and divisor < 0:
                fracs[i] = -1
            else:
                fracs[i] = sol / divisor

        row = get_best(fracs, lambda a: a >= 0 and a <= 1e9)

        if row is None or row == h - 1:
            return (UNSOLVABLE,)

        basic[row] = col
        row += 1

        normalize_row(mat, row, col)

        for i in range(h):
            if i == row:
                continue
            add_row(mat, row, i, -mat[i, col])

        solcell = mat[0, -1]
        if not np.isfinite(solcell) or solcell >= 1e9:
            return (UNBOUNDED,)

    return SOLVED, mat, basic


def run_simplex(objective, mat, rhs_values, eps):
    obj = objective
    nconstraints, nvars = mat.shape
    _mat = mat

    mat = np.matrix([[0.0] * (nvars + 1) for i in range(nconstraints + 1)])
    for i in range(nvars):
        mat[0, i] = -obj[i, 0]
    for i in range(nconstraints):
        for j in range(nvars):
            mat[i + 1, j] = _mat[i, j]
    for i in range(1, nconstraints + 1):
        mat[i, -1] = rhs_values[i - 1, 0]

    solver_state, *results = simplex(mat, eps)

    if solver_state != SOLVED:
        print("The method is not applicable!")
        return

    mat, basic = results

    vals = np.zeros(nvars)

    for i in range(len(basic)):
        if basic[i] < nvars:
            vals[basic[i]] = mat[i + 1, -1]

    print('x*:', vectorAsTuple(vals))
    print("Max value:", round(mat[0, -1], 3))


def interior_solve(objective, constraints, initialsol, alpha, eps):
    c = objective
    a = constraints
    x = initialsol
    solved = False
    eye = np.eye(c.shape[0])
    onevector = np.matrix('; '.join('1' * x.shape[0]), float)
    objval = (x.transpose() * c).A1[0]
    while not solved:
        diag = np.diag(x.A1)
        a_ = a * diag
        c_ = diag * c
        a_t = a_.transpose()
        try:
            p = eye - a_t * inv(a_ * a_t) * a_
        except np.linalg.LinAlgError:
            return (UNSOLVABLE,)
        cp = p * c_
        nu = max(-i for i in cp.A1)
        if nu <= 0:
            return (UNSOLVABLE,)
        x = diag * (onevector + alpha / nu * cp)
        newobjval = (x.transpose() * c).A1[0]
        diff = newobjval - objval
        objval = newobjval
        if diff < eps:
            solved = True
        if objval > 1e9:
            return (UNBOUNDED,)
    return SOLVED, x


def find_max(objective, x):
    return round((objective.T * x).A1[0], 3)


def check_applicability(answer):
    if answer[0] == UNBOUNDED:
        print("The problem does not have solution!")
        return False
    if answer[0] == UNSOLVABLE:
        print("The method is not applicable!")
        return False
    return True


def main():
    inp = input('Vector of coefficients of objective function: ')
    objective = np.matrix(inp, float).T

    constraint_count = int(input('Input number of constraints: '))
    print('Matrix of coefficients of constraints: ')
    in_constraints = ';'.join(input() for i in range(constraint_count))

    constraints = np.matrix(in_constraints, float)

    inp = input('Initial starting point: ')
    initialsol = np.matrix(inp, float).T

    inp = input('Vector of right-hand side numbers: ')
    right_hand_side = np.matrix(inp, float).T

    eps = float(input("Epsilon: "))

    print()
    print('Interior-point method:')

    error = constraints * initialsol - right_hand_side
    if max(abs(i) for i in error.A1) > eps:
        print('The vector', vectorAsTuple(initialsol), 'is not a solution!')
    else:
        for alpha in (0.01, 0.2, 0.5, 0.9):
            print('Alpha =', alpha, end=':\n')
            answer = interior_solve(objective, constraints, initialsol, alpha,
                                    eps)

            if check_applicability(answer):
                print('x* =', vectorAsTuple(answer[1]))
                print('max:', find_max(objective, answer[1]))

    print('Simplex method:')
    run_simplex(objective, constraints, right_hand_side, eps)


if __name__ == "__main__":
    main()
