# %%

import matplotlib.pyplot as plt  # ignore
import numpy as np  # ignore

# %%

data = np.random.normal(size=100)


# %%

plt.hist(data, bins=20, color="red", edgecolor="black")
plt.title("High Granularity Histogram")
plt.show()

# %%

# %%
plt.hist(data, bins=10, density=True, color="red", edgecolor="black")
plt.title("Histogram Example")
plt.show()

# %%
