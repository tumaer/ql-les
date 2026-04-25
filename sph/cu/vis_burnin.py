import matplotlib.pyplot as plt
import numpy as np

u = np.load("res/kolm2d/traj_19/u_512_04500_burnin.npy")
u_mag = np.linalg.norm(u, axis=0)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(u_mag, origin="lower", vmin=0, vmax=6)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
for spine in ax.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_position([0, 0, 1, 1])
fig.savefig("res/burnin_umag.png", dpi=300, bbox_inches="tight", pad_inches=0)
