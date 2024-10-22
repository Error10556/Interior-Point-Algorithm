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
    return SOLVED, x


def find_max(objective, x):
    n = len(objective)
    ans = 0
    for i in range(n):
        ans += objective[i]*x[i]
    return ans


def display(vertical_vec, n):
    s = ''
    for i in range(n):
        s += vertical_vec[i, 0] + ' '
    print(s)
    return s


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
    objective = np.matrix(inp).T
    n = len(inp.split())

    constraint_count = int(input('Input number of constraints: '))
    print('Matrix of coefficients of constraints: ')
    in_constraints = ';'.join(input() for i in range(constraint_count))

    constraints = np.matrix(in_constraints)
    nvars = constraints.shape[1]

    inp = input('Initial starting point: ')
    initialsol = np.matrix(inp).T

    inp = input('Vector of right-hand side numbers: ')
    right_hand_side = ';'.join(input() for i in range(constraint_count))

    eps = int(input())
    alpha05 = 0.5
    alpha09 = 0.9

    answer_alpha05 = interior_solve(objective, constraints, initialsol, alpha05, eps)
    if check_applicability(answer_alpha05):
        print('Alpha = 0.5: x* =', display(answer_alpha05[1], n))
        print('max:', find_max(objective, answer_alpha05[1]))

    answer_alpha09 = interior_solve(objective, constraints, initialsol, alpha09, eps)
    if check_applicability(answer_alpha09):
        print('Alpha = 0.9: x* =', display(answer_alpha09[1], n))
        print(find_max(objective, answer_alpha09[1]))

    run_simplex(objective, nvars, constraints, right_hand_side, eps)


if __name__ == "__main__":
    main()
