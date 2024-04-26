# coding: utf-8
# MatrixTools: FUGUO, 2023/12/06
import argparse
import sys
import numpy as np
import os


def vec_norm(v):
    '''
    :param v: a vector of shape (n,) or (1,n) or (n,1)
    :return: Euclidean norm of v
    '''
    return np.sqrt(np.sum(v * v))


def inner_product(u, v):
    '''
    :param u: a vector of shape (n,) or (1,n) or (n,1)
    :param v: a vector of the same shape as u
    :return: inner product between u and v
    '''
    return np.sum(u * v)


def round_off_to_zeros(A):
    '''
    :param A: a matrix or a vector
    :return: A', where small elements to scale 1e-8 are rounded off to zeros
    '''
    A = np.where(np.abs(A) < 1e-8, 0, A)
    return A


def equal_to_zero(v):
    '''
    Return whether the round-off value of v is 0
    :param v: an int or float value
    :return: True if round-off value of v equals 0, otherwise False
    '''
    return np.abs(v) < 1e-8


def PLU(A):
    '''
    Calculate PLU decomposition of A such that A=PLU, where P is a permutation matrix, L is
    lower-triangular, U is upper-triangular. Time complexity: O(n^3).
    :param A: shape (n,n)
    :return: P, L, U, which are all square
    '''
    assert A.shape[0] == A.shape[1], "A must be square to have PLU decomposition.\n"
    n = A.shape[0]
    U = np.array(A, dtype=float)
    row_numbers = np.arange(n)  # 记录行号的向量
    for i in range(n):
        pi = np.argmax(np.abs(U[i:, i])) + i  # pivot目前在第pi行
        U[i], U[pi] = U[pi].copy(), U[i].copy()  # 交换第i行和第pi行, 新的主元位置在第i行第i列
        row_numbers[i], row_numbers[pi] = row_numbers[pi], row_numbers[i]
        if U[i][i] != 0:
            for j in range(i+1, n):  # 消去第j行主元下面的元素
                coe = -U[j][i]/U[i][i]
                U[j][i:n] += coe * U[i][i:n]
                U[j][i] = -coe

    L = np.tril(U)  # L为结果的下三角矩阵, 且对角线元素为1
    for i in range(n):
        L[i][i] = 1.0
    U = np.triu(U)  # U为结果的上三角矩阵
    U = round_off_to_zeros(U)
    P = np.zeros(shape=(n,n), dtype=int)
    for i in range(n):
        P[i][row_numbers[i]] = 1
    P = P.T

    return P, L, U


def row_echelon(A):
    '''
    Reduce A into row echelon form with partial pivoting
    :param A: shape (m,n)
    :return: R, pivots: where R is row echelon form of A, and pivots is a list of pivots' positions
    '''
    m = A.shape[0]
    n = A.shape[1]
    R = np.array(A)
    pivot_positions = []  # 记录主元的位置
    cur_row = 0
    for j in range(n):
        pi = np.argmax(np.abs(R[cur_row:, j])) + cur_row  # pivot目前在第pi行
        if not equal_to_zero(R[pi][j]):
            R[pi], R[cur_row] = R[cur_row].copy(), R[pi].copy()  # 交换第pi行和第cur_row行, 新主元位置在第cur_row行第j列
            pivot_positions.append((cur_row, j))
            for i in range(cur_row+1, m):  # 消去主元下面的元素
                coe = -R[i][j]/R[cur_row][j]
                R[i][j:] += coe * R[cur_row][j:]
            cur_row += 1
            if cur_row >= m:
                break

    R = round_off_to_zeros(R)
    return R, pivot_positions


def reduced_row_echelon(A):
    '''
    Reduce A into reduced row echelon form with Gauss-Jordan method and partial pivoting.
    :param A: shape (m,n)
    :return: R, pivots: where R is the reduced row echelon form of A, and pivots is a list of pivots' positions
    '''
    m = A.shape[0]
    n = A.shape[1]
    R = np.array(A)
    pivot_positions = []  # 记录主元的位置
    cur_row = 0
    for j in range(n):
        pi = np.argmax(np.abs(R[cur_row:, j])) + cur_row  # pivot目前在第pi行
        if not equal_to_zero(R[pi][j]):
            R[pi], R[cur_row] = R[cur_row].copy(), R[pi].copy()  # 交换第pi行和第cur_row行, 新主元位置在第cur_row行第j列
            pivot_positions.append((cur_row, j))
            for i in range(cur_row + 1, m):  # 消去主元下面的元素
                coe = -R[i][j] / R[cur_row][j]
                R[i][j:] += coe * R[cur_row][j:]
            cur_row += 1
            if cur_row >= m:
                break

    R = round_off_to_zeros(R)

    r = len(pivot_positions)  # r既是主元的个数, 也是矩阵的秩
    for i in range(r):
        j = pivot_positions[i][1]  # (i,j)是主元的位置
        R[i] = R[i] / R[i][j]  # 将所有的主元变成1

    for i in range(r):
        j = pivot_positions[i][1]  # (i,j)是主元的位置
        for t in range(i):
            R[t] += (-R[t][j]) * R[i]  # 消去主元位置(i,j)正上方的所有元素

    R = round_off_to_zeros(R)
    return R, pivot_positions


def solve_equation(C):
    '''
    Solve Ax=b by calculating a particular solution to Ax=b and general solutions to Ax=0.
    :param C: the augmented matrix of shape (m,n+1), which equals [A|b]
    :return: (ps, gs_list) or "nosolution":
            where ps is a particular solution to Ax=b,
            and gs_list is a list of general solutions to Ax=0.
            ps and every solution in gs_list has shape of (n,1).
            The general solution form of Ax=b is $x=ps + \sum_i{\alpha_i * gs_i}$
    '''
    m = C.shape[0]
    n = C.shape[1] - 1
    assert n > 0, "The augment matrix (input) must have 2 or more columns"
    C, pivot_positions = reduced_row_echelon(C)  # 将C转化为Reduced Row Echelon形式, 并得到主元位置列表
    r = len(pivot_positions)  # rank of C
    if r > 0 and pivot_positions[-1][1] == n:  # 如果增广矩阵最后一列是主元列, 则方程无解
        return "nosolution"

    pivot_cols = [pivot_pos[1] for pivot_pos in pivot_positions]  # 主元所在的列
    non_pivot_cols = [i for i in range(n) if i not in pivot_cols]  # 非主元所在的列(不含最后一列)

    A = C[:, :n]  # 系数矩阵 shape (m,n)
    b = C[:, n:n+1]  # 向量 shape (m,1)

    # 得到Ax=b的一个特解ps
    ps = np.zeros(shape=(n,1), dtype=float)
    for i in range(r):
        index = pivot_cols[i]
        ps[index][0] = b[i][0]

    # 得到Ax=0的一组通解gs_list
    gs_list = []
    for i in non_pivot_cols:
        gs = np.zeros(shape=(n,1), dtype=float)
        gs[i][0] = -1  # 将某个非主元列对应的解分量设置为1
        v = A[:, i:i+1]  # v是系数矩阵的第i列
        for t in range(r):
            index = pivot_cols[t]
            gs[index][0] = v[t][0]
        gs_list.append(gs)

    return ps, gs_list


def rank(A):
    '''
    :param A: a matrix of shape (m,n)
    :return: the rank of A
    '''
    _, pivot_postions = row_echelon(A)
    r = len(pivot_postions)  # 矩阵的秩就是行阶梯型的主元个数
    return r


def determinant(A):
    '''
    :param A: an (n,n) matrix
    :return: the determinant of A
    '''
    assert A.shape[0] == A.shape[1], "A must be square to have a determinant.\n"
    n = A.shape[0]

    P, L, U = PLU(A)
    det_U = np.prod(U.diagonal())  # 计算U的行列式

    inv_cnt = 0  # 记录置换P的逆序对个数
    vp = np.zeros(shape=(n,), dtype=int)
    for i in range(n):
        for j in range(n):
            if P[i][j] == 1:
                vp[i] = j
    for i in range(n):
        for j in range(n):
            if vp[i] > vp[j]:
                inv_cnt += 1
    det_P = 1 if inv_cnt % 2 == 0 else -1  # 如果P是偶置换, 行列式为1; 如果是奇置换, 行列式为-1

    det = det_P * det_U
    det = round_off_to_zeros(det)
    return det


def QR_Gram_Schmidt(A):
    '''
    Calculate QR decomposition of A with modified Gram-Schmidt algorithm.
    :param A: an (m,n) matrix with linearly independent columns
    :return: (Q, R), where Q is an (m,n) matrix with orthonormal columns, R is an (n,n) upper-triangular matrix.
    '''
    m = A.shape[0]
    n = A.shape[1]
    Q = np.array(A)
    for j in range(n):
        col_norm = vec_norm(Q[:, j])  # 得到第j列的模
        if not equal_to_zero(col_norm):
            Q[:, j] /= col_norm  # 将第j列归一化
            vj = np.array(Q[:, j]).reshape((m, 1))  # vj是矩阵第j列, shape (m,1)
            if j < n-1:
                Q[:, j+1:] -= vj @ (vj.T @ Q[:, j+1:])  # 利用第j列对剩余的列更新

    Q = round_off_to_zeros(Q)
    R = Q.T @ A
    R = round_off_to_zeros(R)
    return Q, R


def Householder_reduction(A):
    '''
    Calculate QR decomposition of A with Householder reduction.
    :param A: an (m,n) matrix
    :return: (Q, R), where Q is an (m,m) orthonormal matrix, R is an (m,n) upper-trapezoidal matrix.
    '''
    m = A.shape[0]
    n = A.shape[1]
    k = m if m < n else n
    R = np.array(A)
    P = np.eye(m, dtype=float)  # P用来记录正交约简累积的结果

    for i in range(k):
        vi = R[i:, i:i+1]  # vi是第i列自第i行之后的部分, shape (n-i,1)
        vi_norm = vec_norm(vi)
        if equal_to_zero(vi_norm):  # 当vi全为0的时候, 直接进行下一轮循环
            continue
        new_vi = np.zeros_like(vi, dtype=float)
        new_vi[0][0] = vi_norm
        u = new_vi - vi
        if equal_to_zero(vec_norm(u)):  # 当u全为0是, 说明vi=new_vi, 不需要更新
            continue
        u /= vec_norm(u)
        R[i:] -= 2 * u @ (u.T @ R[i:])  # 只对自第i行下面的部分更新, 用矩阵×向量替代矩阵×矩阵, 提高计算效率
        P[i:] -= 2 * u @ (u.T @ P[i:])

    Q = P.T  # PA=R => A=QR
    R = round_off_to_zeros(R)
    return Q, R


def Givens_reduction(A):
    '''
    Calculate QR decomposition of A with Givens reduction.
    :param A: an (m,n) matrix
    :return: (Q, R), where Q is an (m,m) orthonormal matrix, R is an (m,n) upper-trapezoidal matrix.
    '''
    m = A.shape[0]
    n = A.shape[1]
    k = m if m < n else n
    R = np.array(A)
    P = np.eye(m, dtype=float)  # P用来记录正交约简累积的结果

    for i in range(k):
        for j in range(i+1, m):
            if equal_to_zero(R[j][i]):
                continue
            xi = R[i][i]
            xj = R[j][i]
            c = xi / np.sqrt(xi ** 2 + xj ** 2)
            s = xj / np.sqrt(xi ** 2 + xj ** 2)
            # 根据c,s对原矩阵的i,j行进行旋转操作
            Ri = np.array(R[i]).reshape((1, n))
            Rj = np.array(R[j]).reshape((1, n))
            R[i] = c * Ri + s * Rj
            R[j] = -s * Ri + c * Rj
            # 根据c,s对矩阵P的i,j行进行旋转操作
            Pi = np.array(P[i]).reshape((1, m))
            Pj = np.array(P[j]).reshape((1, m))
            P[i] = c * Pi + s * Pj
            P[j] = -s * Pi + c * Pj

    Q = P.T  # PA=R => A=QR
    R = round_off_to_zeros(R)
    return Q, R


def URV(A):
    '''
    Calculate URV decomposition of A (i.e. A=U @ R @ V^T)
    :param A: an (m,n) matrix
    :return: (U, R, V), where U is an (m,m) orthonormal matrix, R is an (m,n) matrix,
            and V is an (n,n) orthonormal matrix.
    '''
    def orthonormal_basis_of_range(A):
        '''
        :param A: an (m,n) matrix
        :return: an (m,r) matrix whose columns are orthonormal basis of R(A)
        '''
        m = A.shape[0]
        n = A.shape[1]

        Q, _ = QR_Gram_Schmidt(A)  # Q与A形状相同, 除了全为0的列外, 其余列是A的一组标准正交基
        is_zero_col = [equal_to_zero(vec_norm(Q[:, i])) for i in range(n)]
        r = n - (np.array(is_zero_col)).sum()  # r为非零列的个数

        # 去除Q中的零列, 得到矩阵new_Q
        new_Q = np.zeros(shape=(m,r), dtype=float)
        cur_col = 0
        for i in range(n):
            if not is_zero_col[i]:
                new_Q[:, cur_col] = Q[:, i]
                cur_col += 1

        return new_Q

    def orthonormal_basis_of_nullspace(A):
        '''
        :param A: an (m,n) matrix
        :return: an (m,n-r) matrix, whose columns are orthonormal basis of N(A) where r is the rank of A
        '''
        m = A.shape[0]
        n = A.shape[1]

        C = np.zeros(shape=(m, n+1), dtype=float)
        C[:, :n] = A[:, :]  # C是方程Ax=0的增广矩阵
        ps, gs_list = solve_equation(C)  # 方程一定有解, 且特解为0. ps, gs_list分别为特解和通解
        if len(gs_list) == 0:  # 当通解维数为0时, 返回(n,0)矩阵
            return np.zeros(shape=(n, 0), dtype=float)
        solutions = np.concatenate(gs_list, axis=1)
        orthonormal_solutions = orthonormal_basis_of_range(solutions)
        return orthonormal_solutions

    orth_basis_RA = orthonormal_basis_of_range(A)  # R(A)的标准正交基
    orth_basis_NA = orthonormal_basis_of_nullspace(A)  # N(A)的标准正交基
    orth_basis_RAT = orthonormal_basis_of_range(A.T)  # R(A.T)的标准正交基
    orth_basis_NAT = orthonormal_basis_of_nullspace(A.T)  # N(A.T)的标准正交基
    U = np.concatenate((orth_basis_RA, orth_basis_NAT), axis=1)
    V = np.concatenate((orth_basis_RAT, orth_basis_NA), axis=1)
    R = U.T @ A @ V  # 由 A=U @ R @ V.T 得到
    R = round_off_to_zeros(R)
    return U, R, V


if __name__ == "__main__":
    # parse arguments
    args = {}
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--option", "-opt", "--opt")
        parser.add_argument("--input", "-i")
        parser.add_argument("--output", "-o")
        args = vars(parser.parse_args())
        assert (all(value is not None for value in args.values()))
        args["option"] = args["option"].lower()
        # print(args)
    except Exception as e:
        print(f"Error: Illegal command format\n")
        exit(1)

    # load input
    A = np.loadtxt(args["input"], dtype=float, ndmin=2)

    # dirty work
    np.set_printoptions(precision=5, suppress=True)
    output_dirname = os.path.dirname(args["output"])
    if output_dirname != "":
        os.makedirs(os.path.dirname(args["output"]), exist_ok=True)
    with open(args["output"], mode="w") as f:
        if args["option"] in ["qr", "gram_schmidt", "qr_gram_schmidt"]:
            f.write("QR decomposition with Gram-Schmidt algorithm, s.t. A=QR.\n\n")
            f.write(f"Input matrix A ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(A)+"\n\n")
            Q, R = QR_Gram_Schmidt(A)
            f.write(f"Q ({Q.shape[0]} x {Q.shape[1]}):\n")
            f.write(str(Q)+"\n\n")
            f.write(f"R ({R.shape[0]} x {R.shape[1]}):\n")
            f.write(str(R) + "\n\n")
            f.write(f"(Check) Product of QR ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(Q @ R) + "\n\n")
        elif args["option"] in ["plu", "lu"]:
            f.write("PLU decomposition with Gram-Schmidt algorithm, s.t. A=PLU.\n\n")
            f.write(f"Input matrix A ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(A) + "\n\n")
            P, L, U = PLU(A)
            f.write(f"P ({P.shape[0]} x {P.shape[1]}):\n")
            f.write(str(P) + "\n\n")
            f.write(f"L ({L.shape[0]} x {L.shape[1]}):\n")
            f.write(str(L) + "\n\n")
            f.write(f"U ({U.shape[0]} x {U.shape[1]}):\n")
            f.write(str(U) + "\n\n")
            f.write(f"(Check) Product of PLU ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(P @ L @ U) + "\n\n")
        elif args["option"] in ["det", "determinant"]:
            f.write("Determinant of matrix A.\n\n")
            f.write(f"Input matrix A ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(A) + "\n\n")
            det = determinant(A)
            f.write(f"The determinant of A is {det}\n")
        elif args["option"] in ["rank", ]:
            f.write("Rank of matrix A.\n\n")
            f.write(f"Input matrix A ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(A) + "\n\n")
            r = rank(A)
            f.write(f"The rank of A is {r}\n")
        elif args["option"] in ["householder", ]:
            f.write("Householder Reduction, s.t. A=QR, where R is the reduced matrix.\n\n")
            f.write(f"Input matrix A ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(A) + "\n\n")
            Q, R = Householder_reduction(A)
            f.write(f"Q ({Q.shape[0]} x {Q.shape[1]}) (Orthonormal):\n")
            f.write(str(Q) + "\n\n")
            f.write(f"R ({R.shape[0]} x {R.shape[1]}) (Reduced matrix):\n")
            f.write(str(R) + "\n\n")
            f.write(f"(Check) Product of QR ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(Q @ R) + "\n\n")
        elif args["option"] in ["givens", ]:
            f.write("Householder Reduction, s.t. A=QR, where R is the reduced matrix.\n\n")
            f.write(f"Input matrix A ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(A) + "\n\n")
            Q, R = Givens_reduction(A)
            f.write(f"Q ({Q.shape[0]} x {Q.shape[1]}) (Orthonormal):\n")
            f.write(str(Q) + "\n\n")
            f.write(f"R ({R.shape[0]} x {R.shape[1]}) (Reduced matrix):\n")
            f.write(str(R) + "\n\n")
            f.write(f"(Check) Product of QR ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(Q @ R) + "\n\n")
        elif args["option"] in ["urv", ]:
            f.write("URV decomposition of A, s.t. A=U @ R @ V.T.\n\n")
            f.write(f"Input matrix A ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(A) + "\n\n")
            U, R, V = URV(A)
            f.write(f"U ({U.shape[0]} x {U.shape[1]}):\n")
            f.write(str(U) + "\n\n")
            f.write(f"R ({R.shape[0]} x {R.shape[1]}):\n")
            f.write(str(R) + "\n\n")
            f.write(f"V ({V.shape[0]} x {V.shape[1]}):\n")
            f.write(str(V) + "\n\n")
            f.write(f"(Check) Product of U, R, V.T, ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(U @ R @ V.T) + "\n\n")
        elif args["option"] in ["equation", "eq"]:
            C = A
            f.write("Solve Ax=b, given the augmented matrix C=[A|b].\n\n")
            f.write(f"Input (augmented) matrix C ({C.shape[0]} x {C.shape[1]}):\n")
            f.write(str(C) + "\n\n")
            solution = solve_equation(C)
            if solution == "nosolution":
                f.write(f"No solution.\n")
            else:
                ps, gs_list = solution
                if len(gs_list) > 0:
                    f.write(f"ps (one particular solution to Ax=b):\n")
                    f.write(str(ps)+"\n\n")
                    f.write(f"gs_list ({len(gs_list)} general solutions to Ax=0):\n\n")
                    for i in range(len(gs_list)):
                        f.write(f"{i}-th gs (index from 0):\n"+str(gs_list[i])+"\n\n")
                    f.write(f"The general solution form: ")
                    f.write(r"s = ps + \sum(k_i * gs_i), where k_i are free variables."+"\n")
                else:
                    f.write(f"The only solution is:\n")
                    f.write(str(ps)+"\n\n")
        elif args["option"] in ["row_echelon"]:
            f.write("Reduce A into row echelon form with partial pivoting.\n\n")
            f.write(f"Input matrix A ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(A)+"\n\n")
            R, _ = row_echelon(A)
            f.write(f"Row echelon form R ({R.shape[0]} x {R.shape[1]}):\n")
            f.write(str(R)+"\n\n")
        elif args["option"] in ["reduced_row_echelon"]:
            f.write("Reduce A into reduced row echelon form with partial pivoting and Gauss-Jordan method.\n\n")
            f.write(f"Input matrix A ({A.shape[0]} x {A.shape[1]}):\n")
            f.write(str(A)+"\n\n")
            R, _ = reduced_row_echelon(A)
            f.write(f"Reduced row echelon form R ({R.shape[0]} x {R.shape[1]}):\n")
            f.write(str(R)+"\n\n")
        else:
            print(f"Error: No option named {args['option']}.\n")
            exit(1)
