# Q2: Verify the solution of the homogeneous system by substitution
import sympy as sp

A = sp.Matrix([
    [1, 2, -1, 3],
    [2, 4, 1, 0],
    [3, 6, 0, 3]
])
t_values = [2, -3]

basis = A.nullspace()
x = sp.zeros(A.cols, 1)
for tv, v in zip(t_values, basis):
    x += tv * v

print("Chosen parameters (t1, t2) =", tuple(t_values))
print("Solution x =", list(x))

print("\nSubstitution in each equation:")
all_zero = True
for i in range(A.rows):
    lhs = sum(A[i, j] * x[j] for j in range(A.cols))
    terms = " + ".join(f"({A[i, j]})({x[j]})" for j in range(A.cols))
    print(f" Equation {i + 1}: {terms} = {lhs}")
    all_zero = all_zero and (lhs == 0)

print("\nA * x =", list(A * x))
for i, v in enumerate(basis, start=1):
    print(f"A * v{i} =", list(A * v))

print("\nVerification successful: every equation is satisfied." if all_zero
      else "\nVerification failed.")
