import matplotlib.pyplot as plt  # ignore

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot([1, 2, 3], [1, 2, 3])
axes[1].scatter([1, 2, 3], [3, 2, 1])
plt.tight_layout()
plt.show()
