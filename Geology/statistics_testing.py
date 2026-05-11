import numpy as np
import os

# --- Paths ---
# folder_path = '/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/repaint_ensemble/final' #unnormalise # folder with 10 2D samples
folder_path = '/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/repaint_ensemble/trial/1000' #normalised samples
output_mean_file = './pixel_mean.npy'
output_std_file  = './pixel_std.npy'

# --- Load 10 2D samples ---
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

samples = []
for f in file_list:
    sample = np.load(os.path.join(folder_path, f))  # shape (H, W)
    sample = np.squeeze(sample)
    samples.append(sample)

# Stack into array: shape (10, H, W)
generated_samples = np.stack(samples, axis=0)
print("Loaded samples shape:", generated_samples.shape)

# --- Compute pixel-wise mean and std ---
pixel_mean = np.mean(generated_samples, axis=0)  # shape (H, W)
pixel_std = np.std(generated_samples, axis=0)    # shape (H, W)
np.save("/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/norpixel_mean_120_1000.npy", pixel_mean)
np.save("/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/norpixel_std_120_1000.npy", pixel_std)


# --- Define mask (same as used during repaint) ---
H, W = pixel_std.shape

mask = np.ones((H, W))
mask[:, 120:] = 0   # right side was generated

known_std = pixel_std[mask == 1]
generated_std = pixel_std[mask == 0]

print("Mean STD (known region):", np.mean(known_std))
print("Mean STD (generated region):", np.mean(generated_std))

print("Max STD (known region):", np.max(known_std))
print("Max STD (generated region):", np.max(generated_std))

print("Mean pixel value (known region):", np.mean(pixel_mean[mask==1]))
print("Mean pixel value (generated region):", np.mean(pixel_mean[mask==0]))

threshold = 0.4  # adjust based on your std values
high_uncertainty_fraction = np.mean(pixel_std > threshold)
print("Fraction of high-uncertainty pixels in generated region:", high_uncertainty_fraction)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# --- define mask ---
H, W = pixel_std.shape
mask = np.ones((H, W))
mask[:, 120:] = 0   # generated region

# generated region mask
generated_region = (mask == 0)

# compute distance from known region boundary
distance_map = distance_transform_edt(generated_region)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# --- define mask ---
H, W = pixel_std.shape
mask = np.ones((H, W))
mask[:, 120:] = 0   # generated region

# generated region mask
generated_region = (mask == 0)

# compute distance from known region boundary
distance_map = distance_transform_edt(generated_region)

# collect std values vs distance
distances = distance_map[generated_region]

pixel_mean_gen = pixel_mean[generated_region]  # shape: (#pixels in generated region,)
pixel_std_gen = pixel_std[generated_region] # shape: (#pixels in generated region,)
mean_vs_distance = []
std_vs_distance = []


for d in range(0, int(distances.max())):
    mask_d = (distances >= d) & (distances < d+1)
    if np.sum(mask_d) > 0:
        mean_vs_distance.append(pixel_mean_gen[mask_d].mean())
        std_vs_distance.append(pixel_std_gen[mask_d].mean())

plt.figure()
plt.plot(mean_vs_distance)
plt.title("Mean Value vs Distance from Boundary")
plt.xlabel("Distance (pixels)")
plt.ylabel("Mean Pixel Value")
plt.savefig(
        "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/normeanconditioning_180_1000.png",
        dpi=150
    )
plt.show()
# collect std values vs distance
distances = distance_map[generated_region]
std_values = pixel_std[generated_region]

# bin distances
bins = np.arange(0, np.max(distances)+1)
mean_std_per_dist = []

for b in bins:
    mask_bin = (distances >= b) & (distances < b+1)
    if np.sum(mask_bin) > 0:
        mean_std_per_dist.append(np.mean(std_values[mask_bin]))
    else:
        mean_std_per_dist.append(np.nan)

# plot
plt.figure(figsize=(6,4))
plt.plot(bins, mean_std_per_dist, marker='o')
plt.xlabel("Distance from mask boundary (pixels)")
plt.ylabel("Average STD")
plt.title("Uncertainty vs Distance from Boundary")
plt.grid(True)
plt.savefig(
        "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/norconditioning_180_1000.png",
        dpi=150
    )
plt.show()




import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.imshow(pixel_mean, cmap='grey')
plt.colorbar()
plt.title('Pixel-wise Mean')
plt.savefig(
        "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/norpixel_mean_180_1000.png",
        dpi=150
    )
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(pixel_std, cmap='hot')
plt.colorbar()
plt.title('Pixel-wise Std (Uncertainty)')
plt.savefig(
        "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/histogram/norpixel_std_180_1000.png",
        dpi=150
    )
plt.show()