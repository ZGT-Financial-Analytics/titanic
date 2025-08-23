# %%

from turtle import color  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401


world_champs = ["Hikaru", "Magnus", "Bobby Fischer"]

chess_championships = [5, 2, 1]
# %%
world_champs = ["Hikaru", "Magnus", "Bobby Fischer"]

chess_championships = [5, 2, 1]
plt.bar(world_champs, chess_championships)
# %%

world_champs = ["Hikaru", "Magnus", "Bobby Fischer"]

chess_championships = [5, 2, 1]

plt.bar(world_champs, chess_championships, color="green")
# %%

world_champs = ["Hikaru", "Magnus", "Bobby Fischer"]

chess_championships = [5, 2, 1]

plt.bar(world_champs, chess_championships, color="green")
plt.xlabel("World Chess Champions Name")
plt.ylabel("Number of Championships Won")

# %%
chess_championships = [5, 2, 1]
world_champs = ["Hikaru", "Magnus", "Bobby Fischer"]
plt.bar(world_champs, chess_championships, color="green")
plt.xlabel("World Chess Champions Name")
plt.ylabel("Number of Championships Won")

# %%
chess_championships = [5, 2, 1]
world_champs = ["Hikaru", "Magnus", "Bobby Fischer"]
plt.bar(world_champs, chess_championships, color="green")
plt.xlabel("World Chess Champions Name")
plt.ylabel("Number of Championships Won")
plt.grid(axis="y")

# %%
# different bar colors
chess_championships = [5, 2, 1]
world_champs = ["Hikaru", "Magnus", "Bobby Fischer"]
colors = ["red", "green", "blue"]
plt.bar(world_champs, chess_championships, color=colors)
plt.xlabel("World Chess Champions Name")
plt.ylabel("Number of Championships Won")
plt.grid(axis="y")
plt.savefig("chesssss_championships.png")
plt.show()
# %%
