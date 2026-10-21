# Q6: Compare the transformed coordinates for different shear factors
import numpy as np
import matplotlib.pyplot as plt

shape = np.array([
    [0, 2, 2, 0],
    [0, 0, 2, 2]
], dtype=float)
factors = [0.5, 1.0, 2.0]

def shear_x(k):
    return np.array([[1, k], [0, 1]])

def shear_y(k):
    return np.array([[1, 0], [k, 1]])

print("Transformed coordinates of the vertices")
for name, make in [("x-axis", shear_x), ("y-axis", shear_y)]:
    print(f"\nShear along {name}")
    print(f"{'k':>5} | {'(0,0)':>10} {'(2,0)':>10} {'(2,2)':>10} {'(0,2)':>10}")
    for k in factors:
        new = make(k) @ shape
        cells = [f"({new[0, i]:.1f},{new[1, i]:.1f})" for i in range(4)]
        print(f"{k:>5} | " + " ".join(f"{c:>10}" for c in cells))

def closed(m):
    return np.hstack([m, m[:, :1]])

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
colors = ["tab:green", "tab:orange", "tab:red"]
for axis, make, title in zip(
    ax, [shear_x, shear_y], ["Shear along x-axis", "Shear along y-axis"]
):
    original = closed(shape)
    axis.plot(original[0], original[1], "k-o", label="Original")
    for k, color in zip(factors, colors):
        transformed = closed(make(k) @ shape)
        axis.plot(transformed[0], transformed[1], "-o",
                  color=color, label=f"k = {k}")
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()

plt.tight_layout()
plt.show()
