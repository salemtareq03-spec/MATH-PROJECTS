# Q3: Simulate a three-state Markov Chain for a given transition matrix
import numpy as np

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])
pi0 = np.array([1.0, 0.0, 0.0])
steps = 10
N = 10000

assert P.shape == (3, 3) and np.allclose(P.sum(axis=1), 1), (
    "P must be a 3x3 stochastic matrix"
)
assert np.all(P >= 0), "Transition probabilities must be non-negative"
assert np.isclose(pi0.sum(), 1) and np.all(pi0 >= 0), (
    "Initial distribution must be a probability distribution"
)

print("Transition matrix P:")
print(P)
print("Initial distribution:", pi0)

print(f"\n{'Step':>4} {'P(S1)':>9} {'P(S2)':>9} {'P(S3)':>9}")
dist = pi0.copy()
for n in range(steps + 1):
    print(f"{n:>4} {dist[0]:9.4f} {dist[1]:9.4f} {dist[2]:9.4f}")
    dist = dist @ P

vals, vecs = np.linalg.eig(P.T)
k = np.argmin(np.abs(vals - 1))
stationary = np.real(vecs[:, k])
stationary = stationary / stationary.sum()
print("\nSteady-state distribution:", np.round(stationary, 4))

rng = np.random.default_rng(seed=1)
state = 0
visits = np.zeros(3)
for _ in range(N):
    state = rng.choice(3, p=P[state])
    visits[state] += 1
print(f"Simulated visit frequencies ({N} steps):", np.round(visits / N, 4))
