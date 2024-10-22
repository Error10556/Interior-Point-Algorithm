import numpy as np
inv = np.linalg.inv


SOLVED = 1
UNSOLVABLE = 2
UNBOUNDED = 3

DEFAULT_EPS = 1e-7


def get_best(array, condition=lambda x: True):
    filtered = [i for i in range(len(array)) if condition(array[i])]
    if not filtered:
        return None
    return min(filtered, key=lambda i: array[i])


def normalize_row(mat, row, col):
    mat[row] /= mat[row][col]


def add_row(mat, source_row, target_row, multiplier):
    mat[target_row] += mat[source_row] * multiplier


def simplex(mat, eps):
    h, w = mat.shape
    fracs = np.zeros(h - 1)
    basic = np.arange(h - 1) + h - 1

    while True:
        col = get_best(mat[0], lambda x: x < -eps)
        if col is None or mat[0][col] > -eps:
            break

        for i in range(h - 1):
            sol = mat[i + 1][-1]
            divisor = mat[i + 1][col]
            if sol >= 0 and sol <= eps and divisor < 0:
                fracs[i] = -1
            else:
                fracs[i] = sol / divisor

        row = get_best(fracs, lambda a: a >= 0 and a <= 1e9)

        if row is None or row == h - 1:
            return UNSOLVABLE

        basic[row] = col
        row += 1

        normalize_row(mat, row, col)

        for i in range(h):
            if i == row:
                continue
            add_row(mat, row, i, -mat[i][col])

        solcell = mat[0][-1]
        if not np.isfinite(solcell) or solcell >= 1e9:
            return UNBOUNDED

    return SOLVED


def run_simplex(objective, nconstraints, mat, rhs_values, eps):
    input_vector = objective
    obj = list(map(float, input_vector.split()))
    nvars = len(obj)

    mat = np.zeros((1 + nconstraints, nvars + nconstraints + 1))
    mat[0, :nvars] = -np.array(obj)

    for i in range(1, nconstraints + 1):
        mat[i, -1] = rhs_values[i - 1]

    basic = []
    solver_state = simplex(mat, eps)

    if solver_state != SOLVED:
        print("The method is not applicable!")
        return

    vals = np.zeros(nvars)

    for i in range(len(basic)):
        if basic[i] < nvars:
            vals[basic[i]] = mat[i + 1, -1]

    print("Simplex - x*: (", end="")
    print(", ".join(map(str, vals)), end=")\n")

    print("Simplex - max value:", mat[0, -1])


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
    return SOLVED, x


def find_max(objective, x):
    n = len(objective)
    ans = 0
    for i in range(n):
        ans += objective[i]*x[i]
    return ans


def display(horizontal_vec, n):
    s = ''
    for i in range(n):
        s += horizontal_vec[i] + ' '
    return s


def main():
    inp = input('Vector of coefficients of objective function: ')
    objective = np.array(inp).T
    n = len(inp.split())

    print('Matrix of coefficients of constraints: ')
    in_constraints = ''
    for i in range(n):
        in_constraints += (i != n - 1) ? input() + '; ' : input()
    print(in_constraints)

    constraints = np.matrix(in_constraints)
    nvars = constraints.shape[1]

    inp = input('Initial starting point: ')
    initialsol = np.array(inp).T

    inp = input('Vector of right-hand side numbers: ')
    right_hand_side = np.array(inp).T

    eps = int(input())
    alpha05 = 0.5
    alpha09 = 0.9

    answer_alpha05 = interior_solve(objective, constraints, initialsol, alpha05, eps)

    if answer_alpha05[0] == UNBOUNDED:
        return "The problem does not have solution!"
    if answer_alpha05[0] == UNSOLVABLE:
        return "The method is not applicable!"

    print('Alpha = 0.5: x* =', answer_alpha05[1])
    print('max:', find_max(objective, display(answer_alpha05[1], n)))

    answer_alpha09 = interior_solve(objective, constraints, initialsol, alpha09, eps)
    print('Alpha = 0.9: x* =', answer_alpha09[1])
    print(find_max(objective, display(answer_alpha09[1], n)))

    run_simplex(objective, nvars, constraints, right_hand_side, eps)


if __name__ == "__main__":
    main()
