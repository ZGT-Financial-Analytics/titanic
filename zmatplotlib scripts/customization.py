# %%
import matplotlib.pyplot as plt


# %%
# color
plt.plot([1, 2, 3], [1, 2, 3], color="green")


# %%
# Dashed line
plt.plot([1, 2, 3], [1, 2, 3], color="green", linestyle="dashed")
# %%
# Dashdot line
plt.plot([1, 2, 3], [1, 2, 3], color="green", linestyle="dashdot")
# %%
# Dotted line
plt.plot([1, 2, 3], [1, 2, 3], color="green", linestyle="dotted")
# %%
plt.plot([1, 2, 3], [1, 2, 3], color="green", linestyle="dashdot")

# %%
plt.plot([1, 2, 3], [1, 2, 3], color="green", linestyle="dashdot", marker="D")

# %%
plt.plot(
    [1, 2, 3], [1, 2, 3], color="green", linestyle="dashdot", marker="D", linewidth=4
)
plt.title("Thick Line, Diamond Markers, Green, dashdot line style")
# %%
