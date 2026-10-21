# Q4: Plot the probability of each state over successive transitions
import numpy as np
import matplotlib.pyplot as plt

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])
pi0 = np.array([1.0, 0.0, 0.0])
steps = 15

assert P.shape == (3, 3) and np.allclose(P.sum(axis=1), 1), (
    "P must be a 3x3 stochastic matrix"
)

history = [pi0]
for _ in range(steps):
    history.append(history[-1] @ P)
history = np.array(history)

vals, vecs = np.linalg.eig(P.T)
k = np.argmin(np.abs(vals - 1))
steady = np.real(vecs[:, k])
steady /= steady.sum()

print(f"{'Step':>4} {'P(S1)':>9} {'P(S2)':>9} {'P(S3)':>9}")
for n, row in enumerate(history):
    print(f"{n:>4} {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f}")
print("\nSteady state:", np.round(steady, 4))

colors = ["tab:blue", "tab:orange", "tab:green"]
plt.figure(figsize=(8, 5))
for i in range(3):
    plt.plot(range(steps + 1), history[:, i], marker="o",
             color=colors[i], label=f"State {i + 1}")
    plt.axhline(steady[i], color=colors[i], linestyle="--", alpha=0.5)
plt.title("Probability of each state over successive transitions")
plt.xlabel("Number of transitions (n)")
plt.ylabel("Probability")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
