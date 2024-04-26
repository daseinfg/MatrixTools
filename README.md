# MatrixTools

UCAS, Matrix Analysis and Applications, 2023

中国科学院大学(国科大) 矩阵分析与应用 2023 大作业

付国 202328014628082

数学公式在GitHub上有显示问题，具体介绍请看README.pdf

## Usage

```
python mt.py --input "path/of/input/txt/file" --option "option-name" --output "path/of/output/txt/file"
```

**Example**

```
python mt.py --input example.txt --option QR --output qr.txt
```

**Alias of arguments**

| Argument   | Alias           |
| ---------- | --------------- |
| `--input`  | `-i`            |
| `--option` | `-opt`, `--opt` |
| `--output` | `-o`            |

**Available options (case free)**

| Options               | Input                            | Function                                                     | Output                                                 | Alias                |
| --------------------- | -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ | -------------------- |
| `QR_Gram_Schmidt`     | $A (m,n)$                        | Calculate QR decoposition of $A$ with modified Gram-Schmidt algorithm. | $Q (m,n)$, $R(n, n)$                                   | `QR`, `Gram_Schmidt` |
| `Householder`         | $A (m,n)$                        | Calculate QR decomposition of $A$ with Householder reduction,  such that $A=QR$, where $R$ is the reduced upper-trapezoidal result. | $Q(m,m)$, $R(m,n)$                                     |                      |
| `Givens`              | $A (m,n)$                        | Calculate QR decomposition of $A$ with Householder reduction, such that $A=QR$, where $R$ is the reduced upper-trapezoidal result. | $Q(m,m)$, $R(m,n)$                                     |                      |
| `PLU`                 | $A(n,n)$                         | Calculate PLU decomposition of $A$ such that $A=PLU$, where $P$ is a permutation matrix, $L$ is lower-triangular, and $U$ is upper-triangular. | $P(n,n)$, $L(n,n)$, $U(n,n)$                           | `LU`                 |
| `determinant`         | $A(n,n)$                         | Calculate the determinant of $A$.                            | det of $A$                                             | `det`                |
| `rank`                | $A(m,n)$                         | Calculate the rank of $A$.                                   | rank of $A$                                            |                      |
| `URV`                 | $A(m,n)$                         | Calculate URV decoposition of $A$, such that $A=URV^T$.      | $U(m,m)$, $R(m,n)$, $V(n,n)$                           |                      |
| `equation`            | $C(m,n+1)$, the augmented matrix | Solve $Ax=b$ for a general form solution. The input $C$ is $[A|b]$. The solution form is $s=ps+\sum_i k_i gs_i$, where $ps$ is a particular solution to $Ax=b$, and $gs_i$ are solutions to $Ax=0$ | `ps`, `gs_list`, where each solution has shape $(n,1)$ | `eq`                 |
| `row_echelon`         | $A(m,n)$                         | Reduce $A$ into row echelon form with partial pivoting.      | $R(m,n)$                                               |                      |
| `reduced_row_echelon` | $A(m,n)$                         | Reduce $A$ into reduced row echelon form with partial pivoting and Gauss-Jordan method. | $R(m,n)$                                               |                      |

**Attention**

+ The input for solving equation $Ax=b$ is the augmented matrix $C=[A|b]$.
+ $A$ must be square to have determinant and PLU decomposition.

+ Difference between `QR_Gram_Schmidt` and `Householder`, `Givens`
    + In `QR_Gram_Schmidt`, you orthogonalize columns of $A$ to get $Q$, so $Q$ has identical shape to $A$ and $R$ is square.
    + In `Householder` and `Givens`, you reduce $A$ to upper-trapezoidal $R$ with orthogonal reduction, so $R$ has identical shape to $A$ and $Q$ is square.
+ The reduced results of `Householder` and `Givens` is upper-trapezoidal, which maybe not in row echelon form.
