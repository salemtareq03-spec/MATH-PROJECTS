# Q1: Solve a homogeneous system A x = 0 and find the dimension of its solution space
import sympy as sp

A = sp.Matrix([
    [1, 2, -1, 3],
    [2, 4, 1, 0],
    [3, 6, 0, 3]
])

n = A.shape[1]
print("Coefficient matrix A =")
sp.pprint(A, use_unicode=False)

rank = A.rank()
nullity = n - rank
print("\nRank of A                 :", rank)
print("Number of unknowns (n)    :", n)
print("Dimension of solution space = n - rank =", nullity)

print("\nRow-reduced echelon form of A:")
sp.pprint(A.rref()[0], use_unicode=False)

basis = A.nullspace()
print("\nBasis of the solution space:")
for i, v in enumerate(basis, start=1):
    print(f" v{i} =", list(v))

t = sp.symbols(f"t1:{nullity + 1}")
x = sp.zeros(n, 1)
for ti, v in zip(t, basis):
    x += ti * v
print("\nGeneral solution (x1, x2, x3, x4) =", tuple(x))
