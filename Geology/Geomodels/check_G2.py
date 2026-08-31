import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# below code to check the number of G2 patches in the folder and print the file at index 100
# g2_folder = (
#     "/Home/siv36/hesal5042/Research/NORCE/"
#     "inPainting_diffusionModel/Geology/"
#     "sliced_data/Additional/XZ_numpy_patches/NOFAULT_MODEL_R2"
# )

# files = sorted(glob.glob(os.path.join(g2_folder, "*.npy")))

# print("Total G2 patches:", len(files))
# print("File at index 100:", files[100])
#output: File at index 100: /Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/sliced_data/Additional/XZ_numpy_patches/NOFAULT_MODEL_R2/patch_1088.npy


#below code to check the shape, min, max of the G2 patch at index 100 and plot it with horizontal lines at rows 10, 32, 54 and a vertical line at column 120
# path = (
#     "/Home/siv36/hesal5042/Research/NORCE/"
#     "inPainting_diffusionModel/Geology/"
#     "sliced_data/Additional/XZ_numpy_patches/"
#     "NOFAULT_MODEL_R2/patch_1088.npy"
# )

# g2 = np.load(path)
# g2 = np.squeeze(g2)

# print("Shape:", g2.shape)
# print("Min:", g2.min())
# print("Max:", g2.max())

# plt.figure(figsize=(12, 4))
# plt.imshow(g2, cmap="gray", aspect="auto")

# for r in [10, 32, 54]:
#     plt.axhline(r, linestyle="--", linewidth=1.5, label=f"Row {r}")

# plt.axvline(120, linestyle="--", linewidth=2, label="Conditioning boundary")

# plt.xlabel("Horizontal pixel")
# plt.ylabel("Row")
# plt.title("Original G2 patch: patch_1088.npy")
# plt.colorbar(label="Porosity")
# plt.legend()
# plt.tight_layout()
# plt.savefig("original_G2_patch_1088.png", dpi=300)
# plt.show()
#output: Shape: (64, 256), Min: 0.0200620182,Max: 0.379999995 visua saved




# Below code to compute the variogram of the G2 patch at index 100 for row 32 +/- 4 rows and plot it
# path = (
#     "/Home/siv36/hesal5042/Research/NORCE/"
#     "inPainting_diffusionModel/Geology/"
#     "sliced_data/Additional/XZ_numpy_patches/"
#     "NOFAULT_MODEL_R2/patch_1088.npy"
# )

# g2 = np.squeeze(np.load(path))

# mask_position = 120
# pixel_size_m = 10

# # Same window as your evaluation: Row 32 +/- 4 rows
# window = g2[28:37, mask_position:]

# lags = np.arange(1, 131)  # 10 m to 1300 m
# gamma = []

# for h in lags:
#     d = window[:, h:] - window[:, :-h]
#     gamma.append(0.5 * np.mean(d ** 2))

# gamma = np.array(gamma)
# lags_m = lags * pixel_size_m

# plt.figure(figsize=(9, 5))
# plt.plot(lags_m, gamma)

# plt.xlabel("Lag distance (m)")
# plt.ylabel("Semivariance")
# plt.title("Original G2 — Row 32 Variogram")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("original_G2_variogram_row32.png", dpi=300)
# plt.show()
#output: Variogram plot saved as original_G2_variogram_row32.png


#below code to plot the region used for the Row-32 variogram calculation
# path = (
#     "/Home/siv36/hesal5042/Research/NORCE/"
#     "inPainting_diffusionModel/Geology/"
#     "sliced_data/Additional/XZ_numpy_patches/"
#     "NOFAULT_MODEL_R2/patch_1088.npy"
# )

# g2 = np.squeeze(np.load(path))

# # Same region used for Row-32 variogram
# window = g2[28:37, 120:]

# plt.figure(figsize=(12, 4))
# plt.imshow(
#     window,
#     cmap="gray",
#     aspect="auto",
#     extent=[0, 1360, 36, 28]
# )

# plt.xlabel("Distance into unknown region (m)")
# plt.ylabel("Row")
# plt.title("G2: region used to calculate Row-32 variogram")
# plt.colorbar(label="Porosity")
# plt.tight_layout()
# plt.savefig("original_G2_variogram_region.png", dpi=300)
# plt.show()


#below code what is being compared at different lag distances and how many pairs remain.
# path = (
#     "/Home/siv36/hesal5042/Research/NORCE/"
#     "inPainting_diffusionModel/Geology/"
#     "sliced_data/Additional/XZ_numpy_patches/"
#     "NOFAULT_MODEL_R2/patch_1088.npy"
# )

# g2 = np.squeeze(np.load(path))

# # Same Row-32 +/- 4 window and unknown region
# window = g2[28:37, 120:]

# for lag_m in [400, 600, 800, 900, 1000, 1100, 1200, 1300]:

#     h = lag_m // 10

#     d = window[:, h:] - window[:, :-h]

#     gamma = 0.5 * np.mean(d**2)

#     print(
#         f"Lag = {lag_m:4d} m | "
#         f"horizontal pairs/row = {window.shape[1] - h:3d} | "
#         f"total pairs = {d.size:4d} | "
#         f"semivariance = {gamma:.8f}"
#     )
#output: Lag =  400 m | horizontal pairs/row =  96 | total pairs =  864 | semivariance = 0.00917639
# Lag =  600 m | horizontal pairs/row =  76 | total pairs =  684 | semivariance = 0.00627747
# Lag =  800 m | horizontal pairs/row =  56 | total pairs =  504 | semivariance = 0.00850476
# Lag =  900 m | horizontal pairs/row =  46 | total pairs =  414 | semivariance = 0.00597799
# Lag = 1000 m | horizontal pairs/row =  36 | total pairs =  324 | semivariance = 0.00079148
# Lag = 1100 m | horizontal pairs/row =  26 | total pairs =  234 | semivariance = 0.00004018
# Lag = 1200 m | horizontal pairs/row =  16 | total pairs =  144 | semivariance = 0.00002095
# Lag = 1300 m | horizontal pairs/row =   6 | total pairs =   54 | semivariance = 0.00001138
# (torchy) bash-4.4$ 

import numpy as np
import matplotlib.pyplot as plt

path = (
    "/Home/siv36/hesal5042/Research/NORCE/"
    "inPainting_diffusionModel/Geology/"
    "sliced_data/Additional/XZ_numpy_patches/"
    "NOFAULT_MODEL_R2/patch_1088.npy"
)

g2 = np.squeeze(np.load(path))

window = g2[28:37, 120:]

h = 110  # 110 pixels = 1100 m

left = window[:, :-h]
right = window[:, h:]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(left, cmap="gray", aspect="auto", vmin=g2.min(), vmax=g2.max())
axes[0].set_title("First locations (x)")
axes[0].set_xlabel("Remaining comparison pixels")
axes[0].set_ylabel("Rows 28–36")

axes[1].imshow(right, cmap="gray", aspect="auto", vmin=g2.min(), vmax=g2.max())
axes[1].set_title("Locations 1100 m away (x + h)")
axes[1].set_xlabel("Remaining comparison pixels")
axes[1].set_ylabel("Rows 28–36")

plt.suptitle("G2 pairs contributing to variogram at 1100 m")
plt.tight_layout()
plt.savefig("original_G2_variogram_pairs_1100m.png", dpi=300)
plt.show()