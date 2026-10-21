# Q7: Directional derivative of f(x, y) = x^2 + y^2 along a given direction
import sympy as sp

x, y = sp.symbols("x y")
f = x**2 + y**2
point = (1, 2)
direction = (3, 4)

grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
print("f(x, y) =", f)
print("Gradient grad f =", list(grad))

grad_p = grad.subs({x: point[0], y: point[1]})
print(f"grad f at {point} =", list(grad_p))

d = sp.Matrix(direction)
norm = sp.sqrt(d.dot(d))
if norm == 0:
    raise ValueError("Direction vector must not be (0, 0).")
u = d / norm
print("\nDirection vector =", direction, " |d| =", norm)
print("Unit vector u =", list(u))

Duf = grad_p.dot(u)
print(f"\nDirectional derivative D_u f at {point} = {Duf} = {float(Duf):.4f}")
