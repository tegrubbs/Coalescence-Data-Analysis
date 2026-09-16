import numpy as np
import matplotlib.pyplot as plt

# Useful for verifying the bubble size distribution of bubbles injected by phasta.

data = np.loadtxt("injected-bubbles.txt")#, usecols=1)
print(np.max(data))
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data , bins=50, edgecolor="black")
ax.set_xlabel("Bubble size (mm)")
ax.set_ylabel("Count")
ax.set_title("Injected Bubble Size Distribution")
plt.tight_layout()
plt.savefig("injected-bubbles-histogram.png", dpi=150)
plt.show()
print(f"n={len(data)}, mean={data.mean()*1e3:.3f} mm, std={data.std()*1e3:.3f} mm")
