import matplotlib.pyplot as plt  # ignore

# %%
# canvas,axes object creation. 1x3 table of subplots.

fig, axes = plt.subplots(1, 3, figsize=(8, 4))


# line plot
axes[0].plot([1, 2, 3], [1, 2, 3])
axes[0].set_title("line plot")
axes[0].set_xlabel("X label")
axes[0].set_ylabel("Y label")

# scatter
axes[1].scatter([1, 2, 3], [3, 2, 1])
axes[1].set_title("scatter")
axes[1].set_xlabel("X label")
axes[1].set_ylabel("Y label")

# histogram
axes[2].hist([1, 2, 1, 2, 1, 2, 3])
axes[2].set_title("hist")
axes[2].set_xlabel("X label")
axes[2].set_ylabel("Y label")

plt.tight_layout()
plt.show()
