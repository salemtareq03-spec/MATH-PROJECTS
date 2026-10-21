# Q5: Shear transformation along the x-axis and the y-axis
import numpy as np
import matplotlib.pyplot as plt

# Vertices of the square: one column per point
shape = np.array([
    [0, 2, 2, 0],
    [0, 0, 2, 2]
], dtype=float)
k = 1.5

Sx = np.array([[1, k], [0, 1]])
Sy = np.array([[1, 0], [k, 1]])

shear_x = Sx @ shape
shear_y = Sy @ shape

print("Shear factor k =", k)
print("\nShear matrix along x-axis:\n", Sx)
print("Shear matrix along y-axis:\n", Sy)
print(f"\n{'Original':>14} {'Shear along x':>16} {'Shear along y':>16}")
for i in range(shape.shape[1]):
    o = f"({shape[0, i]:.1f}, {shape[1, i]:.1f})"
    a = f"({shear_x[0, i]:.1f}, {shear_x[1, i]:.1f})"
    b = f"({shear_y[0, i]:.1f}, {shear_y[1, i]:.1f})"
    print(f"{o:>14} {a:>16} {b:>16}")

def closed(m):
    return np.hstack([m, m[:, :1]])

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for axis, data, title in zip(
    ax, [shape, shear_x, shear_y],
    ["Original", f"Shear along x (k={k})", f"Shear along y (k={k})"]
):
    c = closed(data)
    axis.fill(c[0], c[1], alpha=0.3)
    axis.plot(c[0], c[1], "o-")
    axis.set_title(title)
    axis.set_xlim(-1, 6)
    axis.set_ylim(-1, 6)
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
