# Q8: Compare directional derivatives of f(x, y) = x^2 + y^2 for three directions
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x, y = sp.symbols("x y")
f = x**2 + y**2
point = (1, 2)
directions = {
    "d1 = (0, 1)": (0, 1),
    "d2 = (1, 1)": (1, 1),
    "d3 = (-1, 1)": (-1, 1)
}

grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
g = np.array(grad.subs({x: point[0], y: point[1]}), dtype=float).flatten()
print("f(x, y) =", f, "   Point =", point)
print("grad f at the point =", g, "   |grad f| =", round(float(np.linalg.norm(g)), 4))

print(f"\n{'Direction':>14} {'Unit vector':>20} {'D_u f':>9}")
names, values = [], []
for name, direction in directions.items():
    d = np.array(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError(f"Direction vector for {name} must not be zero.")
    u = d / norm
    val = float(g @ u)
    names.append(name)
    values.append(val)
    print(f"{name:>14}   ({u[0]:7.4f}, {u[1]:7.4f}) {val:9.4f}")

best = names[int(np.argmax(values))]
print("\nFastest increase among the three directions:", best)
print("Maximum possible rate (along grad f) =", round(float(np.linalg.norm(g)), 4))

plt.figure(figsize=(7, 4.5))
bars = plt.bar(names, values, color=["tab:blue", "tab:orange", "tab:green"])
plt.axhline(np.linalg.norm(g), color="red", linestyle="--",
            label="|grad f| (maximum)")
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.08,
             f"{value:.3f}", ha="center")
plt.ylabel("Directional derivative")
plt.title("Directional derivatives of f(x, y) = x^2 + y^2 at (1, 2)")
plt.legend()
plt.tight_layout()
plt.show()
