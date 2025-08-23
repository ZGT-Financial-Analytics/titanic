# %%

from turtle import color  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401

# %%
chess_championships = [5, 2, 1]
world_champs = ["Hikaru", "Magnus", "Bobby Fischer"]
colors = ["red", "green", "blue"]
plt.bar(world_champs, chess_championships, color=colors)
plt.xlabel("World Chess Champions Name")
plt.ylabel("Number of Championships Won")
plt.grid(axis="y")

# annotate time
plt.annotate(
    "Hikaru Nakamura's Record num of wins",
    xy=(0, 5),
    xytext=(2, 5),  # 2 units to the right
    textcoords='data',
    ha='left',
    va='center',
    arrowprops={
        "facecolor": "red",
        "arrowstyle": "->",
        "color": "red",
        "connectionstyle": "arc3,rad=0.2"
    },
)

# %%
chess_championships = [5, 2, 1]
world_champs = ["Hikaru", "Magnus", "Bobby Fischer"]
colors = ["red", "green", "blue"]
plt.bar(world_champs, chess_championships, color=colors)
plt.xlabel("World Chess Champions Name")
plt.ylabel("Number of Championships Won")
plt.grid(axis="y")
plt.annotate(
    "Hikaru Nakamura's Record num of wins",
    xy=(0, 5),
    xytext=(2, 5),  # 2 units to the right
    textcoords='data',
    ha='left',
    va='center',
    arrowprops={
        "facecolor": "red",
        "arrowstyle": "->",
        "color": "red",
        "connectionstyle": "arc3,rad=0.2"
    },
)
# %%
