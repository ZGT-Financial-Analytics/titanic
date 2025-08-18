# %%
import matplotlib.pyplot as plt  # type: ignore

# %%

x = [1, 2, 3]
y = [18, 10, 0]
plt.scatter(x, y, color="red", marker="o")
plt.xlabel("fat kids")
plt.ylabel("Candy")
plt.title("Fat Kids vs. Candy Remaining")
plt.show()
# %% scatter fat kid diff colors
plt.close("all")
plt.scatter(x, y, c=["red", "green", "blue"], marker="s", s=200)
plt.title("Different color for each point")
plt.show()
# %%
plt.close("all")
x = [1, 2, 3]
y = [18, 10, 0]
labels = ["Fat Kid 1", "Fat Kid 2", "Fat Kid 3"]
plt.scatter(x, y, c=["red", "green", "blue"], marker="s", s=200)
plt.title("Fat Kids vs. Candy Remaining")
for i, label in enumerate(labels):
    plt.text(x[i], y[i] + 1, label, ha="center", fontsize=10)
plt.show()
