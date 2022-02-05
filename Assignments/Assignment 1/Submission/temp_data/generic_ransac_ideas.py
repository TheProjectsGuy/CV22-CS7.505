# Making the RANSAC implementation generic
"""
    The RANSAC implementation in the submission may be very specific.
    It can probably be made more generic. This script has snippets
    that act as anchors
"""

# %% Import everything
import numpy as np


# %%
a = [
    np.array([1,2,3]),
    np.array([[4,5,6], [7,8,9], [10, 11, 12]]),
    np.array([[13,14,15], [16,17,18], [19, 20, 21]]),
    np.array([[22,23,24], [25,26,27], [28, 29, 30]]),
]

k = list(z[[0, 2]] for z in a)
print(*k, sep="\n")

# %%
i = np.random.choice(3, 2, replace=False)
print(f"i = {i}")
k = list(z[i] for z in a)
print(*k, sep="\n")

# %%
