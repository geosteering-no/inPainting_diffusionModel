import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

#device
device = "cuda:7" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#unet model - this is same as training
class GeologyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self._block(1, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        self.bottleneck = self._block(512, 512)

        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
        )

        self.dec1 = self._block(512 + 512, 256)
        self.dec2 = self._block(256 + 256, 128)
        self.dec3 = self._block(128 + 128, 64)
        self.dec4 = self._block(64 + 64, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        e4 = self.enc4(self.down(e3))

        bott = self.bottleneck(self.down(e4))
        bott = bott + t_emb

        d1 = self.dec1(torch.cat([self.up(bott), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up(d3), e1], dim=1))

        return self.final(d4)


#loadin model
model = GeologyUNet().to(device)

model.load_state_dict(
    torch.load(
        "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/model/model100.pth",
        map_location=device
    )
)

model.eval()


#DDPM hyperparameters
T = 1000
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


#dataset
#geology xz pacthes dataset
class GeologyNPY(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx]).astype(np.float32)

        arr_min = arr.min()
        arr_max = arr.max()

        # normalize to [-1, 1]
        arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)
        arr_norm = arr_norm * 2 - 1

        return torch.from_numpy(arr_norm).unsqueeze(0), arr_min, arr_max


dataset = GeologyNPY(
    "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/sliced_data/XZ_numpy_patches"
)

print("Dataset size:", len(dataset))


#repaint inpainting
def repaint(model, x0, mask, T, jump_length, jump_n_sample):

    model.eval()
    device = x0.device
    B = x0.size(0)

    x_t = torch.randn_like(x0)

    for t in tqdm(range(T - 1, -1, -1)):

        for u in range(jump_n_sample if (t > 0 and t % jump_length == 0) else 1):

            t_tensor = torch.tensor([t] * B, device=device).long()

            with torch.no_grad():
                pred_noise = model(x_t, t_tensor)

            # known region
            if t > 0:
                noise_k = torch.randn_like(x0)
                x_known = (
                    sqrt_alphas_cumprod[t - 1] * x0 +
                    sqrt_one_minus_alphas_cumprod[t - 1] * noise_k
                )
            else:
                x_known = x0

            # unknown region
            alpha_t = alphas[t]
            alpha_bar = alphas_cumprod[t]

            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar)) * pred_noise
            )

            if t > 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(betas[t])
            else:
                noise = 0

            x_unknown = mean + sigma * noise

            # merge
            x_t_minus_1 = mask * x_known + (1 - mask) * x_unknown

            # repaint jumps
            if u < jump_n_sample - 1 and t > 0:
                noise_r = torch.randn_like(x_t_minus_1)
                x_t = (
                    torch.sqrt(alphas[t - 1]) * x_t_minus_1 +
                    torch.sqrt(1 - alphas[t - 1]) * noise_r
                )
            else:
                x_t = x_t_minus_1

    return x_t


#GENERATE CONDITIONAL ENSEMBLE

# def generate_ensemble_for_single_condition(
#     model,
#     dataset,
#     n_realizations=1000,
#     mask_position=120,
#     save_dir="/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/repaint_results",
# ):

#     os.makedirs(save_dir, exist_ok=True)

#     # fixed conditioning image
#     idx = random.randint(0, len(dataset) - 1)
#     image = dataset[idx][0].unsqueeze(0).to(device)

#     mask = torch.ones_like(image)
#     mask[:, :, :, mask_position:] = 0

#     known_region = image * mask

#     print(f"Fixed conditioning slice index: {idx}")

#     outputs = []

#     for i in range(n_realizations):

#         print(f"Generating realization {i+1}/{n_realizations}")

#         output = repaint(
#             model,
#             known_region,
#             mask,
#             T,
#             jump_length=5,
#             jump_n_sample=20
#         )

#         output_cpu = output.detach().cpu()

#         outputs.append(output_cpu)

#         # save individual sample
#         np.save(
#             f"{save_dir}/sample_{i:03d}.npy",
#             output_cpu.numpy()
#         )

#     ensemble = torch.stack(outputs, dim=0)

#     # # save complete ensemble
#     # np.save(f"{save_dir}/ensemble.npy", ensemble.numpy())

#     print("Saved ensemble shape:", ensemble.shape)

#     return ensemble

def generate_ensemble_for_single_condition(
    model,
    dataset,
    n_realizations=50,
    mask_position=120,
    save_dir="/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/repaint_results",
    fixed_idx_path=None
):

    os.makedirs(save_dir, exist_ok=True)
    #keep track of fixed conditioning index in a text file to ensure same conditioning across runs
    #keep track of existing samples to avoid overwriting and allow continuation of generation if interrupted
    #keep same conditioning image across runs by saving/loading index from text file

    if fixed_idx_path is None:
        fixed_idx_path = os.path.join(save_dir, "fixed_conditioning_index.txt")

    if os.path.exists(fixed_idx_path):
        with open(fixed_idx_path, "r") as f:
            idx = int(f.read().strip())
        print(f"Loaded fixed conditioning slice index: {idx}")
    else:
        idx = random.randint(0, len(dataset) - 1)
        with open(fixed_idx_path, "w") as f:
            f.write(str(idx))
        print(f"Created fixed conditioning slice index: {idx}")

    # image = dataset[idx][0].unsqueeze(0).to(device)
    image, arr_min, arr_max = dataset[idx]
    image = image.unsqueeze(0).to(device)

    arr_min = float(arr_min)
    arr_max = float(arr_max)

    print("Original porosity min:", arr_min)
    print("Original porosity max:", arr_max)

    mask = torch.ones_like(image)
    mask[:, :, :, mask_position:] = 0
    known_region = image * mask


    existing_files = sorted(glob.glob(os.path.join(save_dir, "sample_*.npy")))

    existing_indices = []
    for f in existing_files:
        name = os.path.basename(f)
        try:
            number = int(name.replace("sample_", "").replace(".npy", ""))
            existing_indices.append(number)
        except ValueError:
            pass

    if len(existing_indices) > 0:
        start_i = max(existing_indices) + 1
    else:
        start_i = 0

    print(f"Found {len(existing_indices)} existing samples.")
    print(f"Continuing from sample {start_i:03d}.")

    #continye generation from last saved sample to avoid overwriting and allow continuation if interrupted
    for i in range(start_i, n_realizations):

        print(f"Generating realization {i+1}/{n_realizations}")

        output = repaint(
            model,
            known_region,
            mask,
            T,
            jump_length=5,
            jump_n_sample=20
        )

        output_cpu = output.detach().cpu()

        
        #1.saving raw normalized output [-1, 1]
        # this is the direct output of the model and can be useful for debugging or future reprocessing
        np.save(
            f"{save_dir}/sample_{i:03d}_normalized.npy",
            output_cpu.numpy()
        )

        
        #2.png in [0,1] just for quick view
        img_norm = output_cpu.squeeze().numpy()
        img_display = (img_norm + 1) / 2
        img_display = np.clip(img_display, 0, 1)

        plt.imsave(
            f"{save_dir}/sample_{i:03d}_display.png",
            img_display,
            cmap="gray"
        )

        #2.unnormalizing to physical porosity values using original min/max of the conditioning image
        img_porosity = img_display * (arr_max - arr_min) + arr_min

        np.save(
            f"{save_dir}/sample_{i:03d}_porosity.npy",
            img_porosity
        )

        plt.imsave(
            f"{save_dir}/sample_{i:03d}_porosity.png",
            img_porosity,
            cmap="viridis"
        )

    print("Finished generation.")

    #built final ensemble file from all saved samples to ensure it includes all generated samples even if generation was done in multiple runs
    all_files = sorted(glob.glob(os.path.join(save_dir, "sample_*.npy")))

    all_samples = []
    for f in all_files:
        arr = np.load(f)
        all_samples.append(arr)

    ensemble = np.stack(all_samples, axis=0)

    np.save(os.path.join(save_dir, "ensemble.npy"), ensemble)

    print("Saved final ensemble:", ensemble.shape)

    return ensemble


#
#VARIOGRAn COMPUTATION
#

# def compute_semivariograms(
#     ensemble_path,
#     boundary=120,
#     points_y=[10, 32, 54],
#     half_window=4,
#     pixel_size_m=10,
#     max_lag=60
# ):

#     #load ensemble
#     samples = np.load(ensemble_path)

#     print("Original shape:", samples.shape)

#     #remove singleton dimensions
#     samples = np.squeeze(samples)

#     print("Squeezed shape:", samples.shape)

#     #shape should now be (N, H, W)
#     N, H, W = samples.shape

#     #only region to right of boundary
#     samples_right = samples[:, :, boundary:]

#     max_lag = min(max_lag, samples_right.shape[2] - 1)

#     lags = np.arange(0, max_lag + 1)

#     variograms = {}
#     empirical_vars = {}

#     for py in points_y:

#         y0 = max(0, py - half_window)
#         y1 = min(H, py + half_window + 1)

#         #shape: (N, n_rows, n_cols)
#         window = samples_right[:, y0:y1, :]

#         gamma = []

#         empirical_vars[py] = np.var(window)

#         for h in lags:

#             if h == 0:
#                 gamma.append(0.0)
#                 continue

#             left = window[:, :, :-h]
#             right = window[:, :, h:]

#             diffs = right - left

#             gamma.append(0.5 * np.mean(diffs ** 2))

#         variograms[py] = np.array(gamma)

#     
#     #plotting individual variograms
#     

#     plt.figure(figsize=(8, 5))

#     for py, gamma in variograms.items():

#         plt.plot(
#             lags * pixel_size_m,
#             gamma,
#             marker="o",
#             markersize=3,
#             label=f"row {py}"
#         )

#         plt.axhline(
#             empirical_vars[py],
#             linestyle="--",
#             linewidth=1,
#             alpha=0.7
#         )

#     plt.xlabel("Lag distance (m)")
#     plt.ylabel("Semivariance")
#     plt.title("Experimental Semivariograms")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     
#     #mean variograms
#     

#     all_gamma = np.stack(list(variograms.values()), axis=0)

#     mean_gamma = np.mean(all_gamma, axis=0)
#     std_gamma = np.std(all_gamma, axis=0)

#     mean_var = np.mean(list(empirical_vars.values()))

#     plt.figure(figsize=(8, 5))

#     plt.plot(
#         lags * pixel_size_m,
#         mean_gamma,
#         linewidth=2,
#         label="Mean semivariogram"
#     )

#     plt.fill_between(
#         lags * pixel_size_m,
#         mean_gamma - std_gamma,
#         mean_gamma + std_gamma,
#         alpha=0.2,
#         label="±1 std"
#     )

#     plt.axhline(
#         mean_var,
#         linestyle="--",
#         linewidth=2,
#         label=f"Mean variance ≈ {mean_var:.4f}"
#     )

#     plt.xlabel("Lag distance (m)")
#     plt.ylabel("Semivariance")
#     plt.title("Mean Experimental Semivariogram")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     
#     #numerical sill check
#    

#     print("\nSILLchcking")

#     for py, gamma in variograms.items():

#         sill_estimate = np.mean(gamma[-5:])
#         var = empirical_vars[py]

#         print(f"\nRow {py}")
#         print(f"Empirical variance: {var:.4f}")
#         print(f"Estimated sill: {sill_estimate:.4f}")
#         print(f"Ratio: {sill_estimate/var:.3f}")


if __name__ == "__main__":

    
    #generate ensemble
    

    ensemble = generate_ensemble_for_single_condition(
        model,
        dataset,
        n_realizations=100,
        mask_position=120
    )

    # 
    # #then compute variograms
    # 
    # compute_semivariograms(
    #     ensemble_path="/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/repaint_ensemble/final/ensemble.npy",
    #     boundary=120,
    #     points_y=[10, 32, 54],
    #     half_window=4,
    #     pixel_size_m=10,
    #     max_lag=60
    # )